import numpy as np
import gensim
import gensim.downloader as api
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags, strip_multiple_whitespaces
from gensim.models import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec




def getPretrained(kind):
    leng = 300
    if kind == 'glove':
        model = api.load('glove-wiki-gigaword-300')
    elif kind == 'word2vec':
        model = api.load('word2vec-google-news-300')
    elif kind == 'fasttext':
        model = api.load('fasttext-wiki-news-subwords-300')
    elif kind == 'test':
        model = api.load('glove-twitter-50')
        leng = 50
    else:
        raise NotImplementedError
    model.add('<SOS>', np.random.normal(0, 0.01, leng))
    model.add('<EOS>', np.random.normal(0, 0.01, leng))
    model.add('<UNK>', np.random.normal(0, 0.01, leng))
    return model


class MyCorpus(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as file:
            for line in file:
                line = preprocess_string(
                    line, [lambda x: x.lower(), strip_tags, strip_multiple_whitespaces])
                yield ['<SOS>', *line, '<EOS>']


class callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1

    def on_epoch_begin(self, model):
        print('Started epoch {} \r'.format(self.epoch))

    def on_train_begin(self, model):
        print('Started traininge!')


def makeModel(filename, kind):
    sentences = MyCorpus(filename)
    if kind == 'glove':
        raise NotImplementedError
    elif kind == 'word2vec':
        model = gensim.models.Word2Vec(
            sentences=sentences,
            size=100,
            compute_loss=True,
            callbacks=[callback()])
    elif kind == 'fasttext':
        model = gensim.models.FastText(
            sentences=sentences,
            size=100,
            callbacks=[callback()])
    else:
        raise NotImplementedError
    return model


def getVectorModel(pretrained, filename=None, kind='word2vec'):
    if pretrained:
        return getPretrained(kind)
    else:
        assert filename is not None
        return makeModel(filename, kind)



