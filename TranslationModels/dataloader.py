import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence

from gensim.test.utils import datapath
from gensim.utils import chunkize_serial
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation, strip_tags, strip_multiple_whitespaces
from gensim.models.callbacks import CallbackAny2Vec

from TranslationModels.const_vars import *

from utilities.utils import preprocess_line

class tr_data_loader(object):
    def __init__(self, src_vectorModel, tgt_vectorModel,
                 filesrc, filetgt, batch_size,
                 sos_token, eos_token, unk_token,
                 remove_punctuation=False,
                 isTransformer=True, max_batches=None, keep_chance=0.1):
        self.filesrc = filesrc
        self.filetgt = filetgt
        self.batch_size = batch_size
        self.src_vm = src_vectorModel
        self.tgt_vm = tgt_vectorModel

        self.remove_punctuation = remove_punctuation

        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self.src_sos_token_index = self.src_vm.vocab.get(self.sos_token).index
        self.src_eos_token_index = self.src_vm.vocab.get(self.eos_token).index
        self.src_unk_token_index = self.src_vm.vocab.get(self.unk_token).index

        self.tgt_sos_token_index = self.tgt_vm.vocab.get(self.sos_token).index
        self.tgt_eos_token_index = self.tgt_vm.vocab.get(self.eos_token).index
        self.tgt_unk_token_index = self.tgt_vm.vocab.get(self.unk_token).index

        self.isTransformer = isTransformer
        self.max_batches = max_batches
        self.keep_chance = keep_chance

    def collateTransformer(self, list_of_samples):
        """Merges a list of samples to form a mini-batch.
        Args:
          list_of_samples is a list of tuples (src_seq, tgt_seq):
              src_seq is of shape (src_seq_length)
              tgt_seq is of shape (tgt_seq_length)
        Returns:
          src_seqs of shape (max_src_seq_length, batch_size): Tensor of padded source sequences.
          src_mask of shape (max_src_seq_length, batch_size): Boolean tensor showing which elements of the
              src_seqs tensor should be ignored in computations (filled with PADDING_VALUE).
          tgt_seqs of shape (max_tgt_seq_length+1, batch_size): Tensor of padded target sequences.
        """
        src_samples, tgt_samples = list(zip(*list_of_samples))

        max_src_seq_length = max([s.size(0) for s in src_samples])
        src_out_dims = (max_src_seq_length, len(src_samples))
        src_seqs = src_samples[0].data.new(*src_out_dims).fill_(self.src_eos_token_index)
        src_mask = torch.ones(*src_out_dims, dtype=torch.bool)
        for i, src_tensor in enumerate(src_samples):
            length = src_tensor.size(0)
            src_seqs[:length, i] = src_tensor
            src_mask[:length, i] = False

        max_tgt_seq_length = max([s.size(0) for s in tgt_samples])
        tgt_out_dims = (1 + max_tgt_seq_length, len(tgt_samples))
        tgt_seqs = tgt_samples[0].data.new(*tgt_out_dims).fill_(self.tgt_eos_token_index)
        for i, tgt_tensor in enumerate(tgt_samples):
            length = tgt_tensor.size(0)
            tgt_seqs[0, i] = self.tgt_sos_token_index
            tgt_seqs[1:length + 1, i] = tgt_tensor

        return src_seqs, src_mask, tgt_seqs


    def collateRNN(self, list_of_samples):
        newlst = sorted(list_of_samples, key=lambda x: x[0].shape[0], reverse = True)
        src_seq_lengths = [x[0].shape[0] for x in newlst]
        src_seqs = pad_sequence([x[0] for x in newlst], padding_value = self.src_sos_token_index)
        tgt_seqs = pad_sequence([x[1] for x in newlst], padding_value = self.tgt_sos_token_index)
        return src_seqs, src_seq_lengths, tgt_seqs


    def __iter__(self):
        with open(self.filesrc) as file_src, open(self.filetgt) as file_tgt:
            lst = []
            i = 0
            for linesrc, linetgt in zip(file_src, file_tgt):
                if random.random() > self.keep_chance:
                    continue
                linesrc = preprocess_line(linesrc, remove_punctuation=self.remove_punctuation)
                linetgt = preprocess_line(linetgt, remove_punctuation=self.remove_punctuation)
                
                linesrc = [*linesrc, self.eos_token]
                linetgt = [*linetgt, self.eos_token]

                linesrc_index = []
                for w in linesrc:
                    vw_index = self.src_vm.vocab.get(w)
                    if vw_index is None:
                        linesrc_index.append(self.src_unk_token_index)
                    else:
                        linesrc_index.append(vw_index.index)

                linetgt_index = []
                for w in linetgt:
                    vw_index = self.tgt_vm.vocab.get(w)
                    if vw_index is None:
                        linetgt_index.append(self.tgt_unk_token_index)
                    else:
                        linetgt_index.append(vw_index.index)

                if self.isTransformer:
                    if len(linesrc_index) > MAX_LENGTH:
                        continue
                    if len(linetgt_index) > MAX_LENGTH:
                        continue

                linesrc_index = torch.tensor(linesrc_index).long()
                linetgt_index = torch.tensor(linetgt_index).long()

                lst.append((linesrc_index, linetgt_index))
                i += 1
                if i % self.batch_size == 0:
                    yield self.collateTransformer(lst) if self.isTransformer else self.collateRNN(lst)
                    lst = []
                    if (self.max_batches is not None) and (i > self.max_batches):
                        break








                        





class test_data_loader(object):
    def __init__(self, filesrc, filetgt, output_file, model, batch_size,
                 remove_punctuation=False, max_batches=None, keep_chance=0.1, device = 'cpu'):
        self.filesrc = filesrc
        self.filetgt = filetgt
        self.output_file =  output_file
        self.model = model

        self.batch_size = batch_size
        self.remove_punctuation = remove_punctuation

        self.max_batches = max_batches
        self.keep_chance = keep_chance
        self.device = device


    def __iter__(self):
        with open(self.filesrc) as file_src, open(self.filetgt) as file_tgt:
            printlst = []
            i = 0
            for linesrc, linetgt in zip(file_src, file_tgt):
                if random.random() > self.keep_chance:
                    continue
                linetrans = self.model.translate(linesrc, str_out=True, device=self.device)
                linetgt = preprocess_line(linetgt, remove_punctuation=self.remove_punctuation)
                linetgt = [*linetgt, '<EOS>']

                printlst.append('-' * 30)
                printlst.append('\n')
                printlst.append('>' * 20 + '\tInput')
                printlst.append('\n')
                printlst.append(linesrc)
                printlst.append('\n')
                printlst.append('<' * 20 + '\tCandidate translation')
                printlst.append('\n')
                printlst.append(' '.join(linetrans))
                printlst.append('\n')
                printlst.append('+' * 20 + '\tReference translation')
                printlst.append('\n')
                printlst.append(' '.join(linetgt))
                printlst.append('\n')
                self.output_file.writelines(printlst)
                printlst = []
                yield linetrans, [linetgt]
                i += 1
                if (self.max_batches is not None) and (i >= self.max_batches):
                    break