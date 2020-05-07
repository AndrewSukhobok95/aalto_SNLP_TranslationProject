import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from torchtext.data.metrics import bleu_score

from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation, strip_tags, strip_multiple_whitespaces

import TranslationModels.tr_model_utils as tr
from TranslationModels.dataloader import tr_data_loader, test_data_loader
from TranslationModels.const_vars import *

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import random
import numpy as np
import time
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

class Encoder(nn.Module):
    def __init__(self, embedding_vectors, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_vectors))
        self.gru = nn.GRU(input_size=self.embedding.embedding_dim, hidden_size=hidden_size)

    def forward(self, pad_seqs, seq_lengths, hidden, padding_value):
        embedded = self.embedding(pad_seqs)
        output = pack_padded_sequence(embedded, seq_lengths)
        output, hidden = self.gru(output, hidden)
        output, _ = pad_packed_sequence(output, padding_value = padding_value)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size)
    
    
    
    
class Decoder(nn.Module):
    def __init__(self, embedding_vectors, hidden_size, sos_index):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_vectors))
        self.gru = nn.GRU(input_size=self.embedding.embedding_dim, hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, self.embedding.num_embeddings)
        self.logsoft = nn.LogSoftmax(dim = 2)
        self.sos_index = sos_index

    def forward(self, hidden, pad_tgt_seqs=None, teacher_forcing=False):
        if pad_tgt_seqs is None:
            assert not teacher_forcing, 'Cannot use teacher forcing without a target sequence.'
        
        sosrow = torch.tensor([[self.sos_index] * hidden.shape[1]]).to(hidden.device)
        
        if teacher_forcing:
            embedded = self.embedding(torch.cat((sosrow, pad_tgt_seqs[:-1,:]), 0))
            output = F.relu(embedded)
            output, hidden = self.gru(output, hidden)
            output = self.out(output)
            output = self.logsoft(output)

        else:
            emb = self.embedding(sosrow)
            output = torch.empty(1, hidden.shape[1], self.embedding.num_embeddings).to(hidden.device)
            
            for i in range(MAX_LENGTH if pad_tgt_seqs is None else pad_tgt_seqs.shape[0]):
                emb = F.relu(emb)
                o, hidden = self.gru(emb, hidden)
                o = self.out(o)
                o = self.logsoft(o)
                
                o1 = torch.argmax(o, dim = 2)
                output = torch.cat([output, o], dim = 0)
                emb = self.embedding(o1)
                
            output = output[1:, :, :]

        return output, hidden


class RNNModel():
    def __init__(self, src_vectorModel, tgt_vectorModel,
                 encoder_save_path, decoder_save_path,
                 hidden_size):
        self.encoder = None
        self.decoder = None
        self.src_vm = src_vectorModel
        self.tgt_vm = tgt_vectorModel
        self.encoder_save_path = encoder_save_path
        self.decoder_save_path = decoder_save_path
        self.hidden_size = hidden_size
        try:
            self.load(self.encoder_save_path, self.decoder_save_path)
            print('++ Model loaded!')
        except Exception as e:
            print(e)
            print()

    def train(self, filesrc, filetgt, batch_size=64, iters=2, teacher_forcing_ratio=0.5, max_batches=None, device="cpu", keep_chance = 0.9):

        src_padding_value = self.tgt_vm.vocab.get(SOS_token).index
        tgt_padding_value = self.tgt_vm.vocab.get(SOS_token).index

        if self.decoder is None:
            self.encoder = Encoder(self.src_vm.vectors, hidden_size=self.hidden_size)
        if self.decoder is None:
            self.decoder = Decoder(self.tgt_vm.vectors, hidden_size=self.hidden_size, sos_index=tgt_padding_value)

        self.encoder.to(device)
        self.decoder.to(device)

        optimizerEnc = torch.optim.Adam(self.encoder.parameters())
        optimizerDec = torch.optim.Adam(self.decoder.parameters())

        criterion = nn.NLLLoss(ignore_index = tgt_padding_value)

        trainloader = tr_data_loader(
            src_vectorModel=self.src_vm,
            tgt_vectorModel=self.tgt_vm,
            filesrc=filesrc,
            filetgt=filetgt,
            batch_size=batch_size,
            sos_token=SOS_token,
            eos_token=EOS_token,
            unk_token=UNK_token,
            max_batches=max_batches,
            isTransformer=False,
            keep_chance = keep_chance
        )

        self.encoder.train()
        self.decoder.train()

        start = time.time()
        loss = None

        for epoch in range(iters):
            for i, batch in enumerate(trainloader):
                train_inputs, train_lengths, train_targets = batch
                hidden = self.encoder.init_hidden(len(train_lengths)).to(device)
                train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
                
                #optimizer.zero_grad()
                optimizerEnc.zero_grad()
                optimizerDec.zero_grad()

                output, hidden = self.encoder(train_inputs, train_lengths, hidden, src_padding_value)
                output, hidden = self.decoder(hidden, pad_tgt_seqs = train_targets, teacher_forcing = random.random() < teacher_forcing_ratio)

                del train_inputs
                del train_lengths
                del hidden 

                output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])
                train_targets = train_targets.reshape(-1)
                
                loss = criterion(output, train_targets)

                loss.backward()
                #optimizer.step()
                optimizerEnc.step()
                optimizerDec.step()

                if (i + 1) % 100 == 0:
                    dur = (int) (time.time() - start)
                    print("{0:d} batches done in {1:d}m:{2:d}s".format(i + 1, dur // 60, dur % 60), end = '\r')
                    torch.save(self.encoder.state_dict(), self.encoder_save_path)
                    torch.save(self.decoder.state_dict(), self.decoder_save_path)

                torch.cuda.empty_cache()

            torch.save(self.encoder.state_dict(), self.encoder_save_path)
            torch.save(self.decoder.state_dict(), self.decoder_save_path)

            end = time.time()
            dur = (int) (end - start)
            start = end
            print("Epoch {0:d}: Loss:{1:0.3f}              {2:d}m:{3:d}s".format(epoch + 1, loss.item(), dur // 60, dur % 60))

        torch.save(self.encoder.state_dict(), self.encoder_save_path)
        torch.save(self.decoder.state_dict(), self.decoder_save_path)

    def load(self, encoder_path, decoder_path, device="cpu"):
        self.encoder = Encoder(self.src_vm.vectors, hidden_size=self.hidden_size)
        self.decoder = Decoder(self.tgt_vm.vectors, hidden_size=self.hidden_size, sos_index = self.tgt_vm.vocab.get('<SOS>').index)

        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))

        self.encoder.to(device)
        self.decoder.to(device)

        self.encoder.eval()
        self.decoder.eval()


    def translate(self, src_str, str_out=False, device='cpu'):

        src_padding_value = self.tgt_vm.vocab.get(SOS_token).index
        tgt_padding_value = self.tgt_vm.vocab.get(SOS_token).index

        self.encoder.to(device)
        self.decoder.to(device)

        hidden = self.encoder.init_hidden()
        
        src_seq = tr.convert_src_str_to_index_seq(
            src_str=src_str,
            src_VecModel=self.src_vm
        ).unsqueeze(1)
        lens = np.array([len(src_seq),])

        output, hidden = self.encoder(src_seq, lens, hidden, src_padding_value)
        output, hidden = self.decoder(hidden, teacher_forcing = False)
        
        output = torch.argmax(output, dim=2)
        
        try:
            output = output[:(output == self.tgt_vm.vocab.get(EOS_token).index).nonzero()[0][0] + 1]
        except:
            pass

        if str_out:
            output = tr.convert_tgt_index_seq_to_str(output, self.tgt_vm)

        return output


    def eval(self, filesrc, filetgt, output_file_name, batch_size=64, max_batches=None, device="cpu", keep_chance = 0.9):
        if self.encoder is None or self.decoder is None:
            print('Model not loaded!')
            return

        self.encoder.to(device)
        self.decoder.to(device)

        with open(output_file_name, 'w') as output_file:
            testloader = test_data_loader(
                filesrc=filesrc,
                filetgt=filetgt,
                output_file=output_file,
                model = self,
                batch_size=batch_size,
                max_batches=max_batches,
                keep_chance = keep_chance,
                device = device
            )

            self.encoder.eval()
            self.decoder.eval()

            start = time.time()
            
            scores = []
            i = 0
            for batch_candidate, batch_references in testloader:
                cur_score = sentence_bleu(batch_references, batch_candidate, smoothing_function = SmoothingFunction().method3)
                scores.append(cur_score)
                i += 1
                print('', file = output_file)
                print('Sample {0:d}, BLEU score: {1:0.4f}'.format(i, cur_score))
                print('', file = output_file)
                print('Sample {0:d}, BLEU score: {1:0.4f}'.format(i, cur_score), file = output_file)
                print('', file = output_file)
                print('=' * 30, file = output_file)
                print('', file = output_file)

            print('=' * 50, file = output_file)
            print('', file = output_file)
            print('= ' * 25, file = output_file)
            print('', file = output_file)
            print('Average BLEU score: {0:0.4f}, minimum score: {1:0.4f}, maximum score: {2:0.4f}, median score: {3:0.4f}'.format(np.mean(scores), min(scores), max(scores), np.median(scores)), file = output_file)
            print('Average BLEU score: {0:0.4f}, minimum score: {1:0.4f}, maximum score: {2:0.4f}, median score: {3:0.4f}'.format(np.mean(scores), min(scores), max(scores), np.median(scores)))

        return scores