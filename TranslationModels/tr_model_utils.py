import math

import torch
import torch.nn as nn

from TranslationModels.const_vars import *
from utilities.utils import preprocess_line



class PositionalEncoding(nn.Module):
    """This implementation is the same as in the Annotated transformer blog post
        See https://nlp.seas.harvard.edu/2018/04/03/attention.html for more detail.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        assert (d_model % 2) == 0, 'd_model should be an even number.'
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class NoamOptimizer:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()



def convert_src_str_to_index_seq(src_str, src_VecModel, remove_punctuation=False):
    src_unk_token_index = src_VecModel.vocab.get(UNK_token).index

    linesrc = preprocess_line(src_str, remove_punctuation=remove_punctuation)
    # linesrc = preprocess_string(src_str, [strip_punctuation, strip_tags, strip_multiple_whitespaces])
    linesrc = [*linesrc, EOS_token]

    linesrc_index = []
    for w in linesrc:
        vw_index = src_VecModel.vocab.get(w)
        if vw_index is None:
            linesrc_index.append(src_unk_token_index)
        else:
            linesrc_index.append(vw_index.index)

    src_seq = torch.tensor(linesrc_index).long()
    return src_seq


def convert_tgt_index_seq_to_str(tgt_seq, tgt_VecModel):
    tgt_str_list = [tgt_VecModel.index2word[x] for x in tgt_seq]
    return tgt_str_list
