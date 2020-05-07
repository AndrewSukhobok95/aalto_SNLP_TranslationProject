import numpy as np
import pickle as p
import os
import sys
from utilities.utils import language_map


class Embedding:
    def __init__(self, lang, type="w2v", dim=100, max_vocab=50000):
        self.vector_dic = dict()
        self.filename = ''
        self.pickle_filename = ''

        lang_full, lang_short = language_map(lang)
        self.lang_full = lang_full
        self.lang_short = lang_short
        self.max_vocab = max_vocab
        self.dim = dim

        self.dir = "vector_models/" + lang_full

        if type == "w2v":
            self.get_w2v_embeddings()

        if type == "glove":
            self.get_glove_embeddings()

        if type == "ft":
            if dim != 300:
                print("embedding.dim set to 300; only dimension available!")
                self.dim = 300

            self.get_ft_embeddings()

        test_vec = next(iter(self.vector_dic.values()))
        if test_vec.size != dim:
            print("Vector size different than desired dim = ", dim, file=sys.stderr)
            
        if "SOS" not in self.vector_dic:
            self.add_special_words()

    def save_pickle(self):
        if self.vector_dic:
            with open(self.dir + "/" + self.pickle_filename, "wb") as f:
                p.dump(self.vector_dic, f)

    def find_oov_word(self, oov_word):
        with open("vector_models/" + self.lang_full + "/" + self.filename) as f:
            _ = f.readline().split(' ')
            for line in f.readlines():
                line = line.rstrip()
                line = line.split(' ')
                if line[0] == oov_word:
                    vec = [float(x) for x in line[1:]]
                    vec = np.array(vec)
                    self.vector_dic[oov_word] = vec
                    self.save_pickle()
                    return vec

        return self.vector_dic["UNK"]

    def __getitem__(self, words):
        if type(words) == str:
            words = [words]

        embeddings = []

        for word in words:
            if word in self.vector_dic:
                embeddings.append(self.vector_dic[word])
            else:
                embeddings.append(self.find_oov_word(word))

        return np.array(embeddings)

    def get_glove_embeddings(self):
        self.pickle_filename = "glove_embedding_d" + str(self.dim) + '.p'

        self.filename = "glove.6B." + str(self.dim) + "d.txt"
        if self.pickle_filename in os.listdir(self.dir):
            self.vector_dic = load_vector_dict(self.dir + "/" + self.pickle_filename)

        else:
            self.vector_dic = read_vector_file(self.filename, self.lang_full, self.max_vocab)
            self.save_pickle()

    def get_ft_embeddings(self):
        self.pickle_filename = "ft_embedding_d" + str(self.dim) + '.p'
        self.filename = "cc." + self.lang_short + ".300.vec"
        if self.pickle_filename in os.listdir(self.dir):
            self.vector_dic = load_vector_dict(self.dir + "/" + self.pickle_filename)

        else:
            self.vector_dic = read_vector_file(self.filename, self.lang_full, self.max_vocab)
            self.save_pickle()

    def get_w2v_embeddings(self):
        self.pickle_filename = "w2v_embedding_d" + str(self.dim) + '.p'
        self.filename = self.lang_short + "wiki_20180420_" + str(self.dim) + "d.txt"
        if self.pickle_filename in os.listdir(self.dir):
            self.vector_dic = load_vector_dict(self.dir + "/" + self.pickle_filename)

        else:
            self.vector_dic = read_vector_file(self.filename, self.lang_full, self.max_vocab)
            self.add_special_words()
            self.save_pickle()

    def get_closest_word(self, query_vector):
        if query_vector.size != self.dim:
            return

        closest_word = ''
        best_distance = 9999
        for word, vector in self.vector_dic.items():
            dist = np.linalg.norm(query_vector - vector)
            if dist < best_distance:
                best_distance = dist
                closest_word = word

        return closest_word

    def add_special_words(self):
        np.random.seed(92)
        self.vector_dic["<SOS>"] = np.random.normal(0, 0.01, self.dim)
        self.vector_dic["<EOS>"] = np.random.normal(0, 0.01, self.dim)
        self.vector_dic["<UNK>"] = np.random.normal(0, 0.01, self.dim)


def read_vector_file(filename, lang_full, max_vocab):
    vector_dic = {}

    path = "vector_models/" + lang_full + "/" + filename

    if not os.path.exists(path):
        print("Embedding file not downloaded! Run download_script.py first", file=sys.stderr)
        return

    with open(path) as f:
        for i in range(max_vocab):
            line = f.readline().rstrip()
            line = line.split(' ')
            if len(line) < 10:
                continue

            word = line[0]
            vec = line[1:]

            if '\n' in line[-1]:
                line[-1] = line[-1][:-2]

            vec = [float(x) for x in vec]
            vector_dic[word] = np.array(vec)

        return vector_dic


def load_vector_dict(path):
    with open(path, "rb") as f:
        return p.load(f)

