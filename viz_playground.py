from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from utilities.utils import language_map
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import FastTextKeyedVectors
import os
import numpy as np

en_words = [["monkey", "dog", "cat", "cow"],
            ["car", "bike", "airplane", "train"],
            ["one", "two", "three", "four"],
            ["amsterdam", "london", "berlin"],
            ["netherlands", "germany", "england"]]

nl_words = [["aap", "hond", "kat", "koe"],
            ["auto", "fiets", "vliegtuig", "trein"],
            ["een", "twee", "drie", "vier"],
            ["amsterdam", "london", "berlin"],
            ["nederland", "engeland", "duitsland"]]

ru_words = [["обезьяна", "собака", "кошка", "корова"],
            ["машина", "велосипед", "самолет", "поезд"],
            ["один", "два", "три", "четыре"],
            ["амстердам", "лондон", "берлин"],
            ["германия", "англия"]]

all_words = {"nl": nl_words, "en": en_words, "ru": ru_words}


def get_model_name():
    dirs = os.listdir("data/vector_models")
    dirs.sort()
    for i, model_name in enumerate(dirs):
        print(i, "\t", model_name)

    ind = int(input("Model:"))
    if ind < len(dirs):
        model_name = dirs[ind]
        return model_name
    else:
        print("Index not valid!")
        return get_model_name()


def visualize_words(wv, words):
    pca = PCA(n_components=2)

    X = wv[words]
    result = pca.fit_transform(X)

    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    plt.show()


def visualize_language():
    model_name = get_model_name()
    path = "data/vector_models/" + model_name

    if "ft" in model_name:
        wv = FastTextKeyedVectors.load(path)
    else:
        wv = KeyedVectors.load_word2vec_format(path, binary=True)

    print("Vocab size:", len(wv.vocab))

    words = [""]
    if "en" in model_name:
        words = en_words

    if "nl" in model_name:
        words = nl_words

    visualize_words(wv, words)


def get_subplot_for_data(lang="en"):
    lang_full, lang_short = language_map(lang)
    fig = plt.figure()

    plot_labels = {"w2v": "Word2Vec", "ft": "FastText",
                   "cbow": "CBOW", "sg": "Skip-Gram"}

    for i, type in enumerate(["w2v", "ft"]):
        for j, hp in enumerate(["cbow", "sg"]):
            print(type, hp)

            # First word2vec
            model_name = type + "_" + lang + "_d100_" + hp + "_st.bin"
            path = "data/vector_models/" + model_name

            if type == "ft":
                wv = FastTextKeyedVectors.load(path)
            else:
                wv = KeyedVectors.load_word2vec_format(path, binary=True)

            words = all_words[lang]

            total_words = []
            for topic in words:
                total_words.extend(topic)

            pca = PCA(n_components=2)

            X = wv[wv.vocab]
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)

            X -= mean
            X /= var
            pca.fit(X)

            # Start subplot
            subplot_num = i * 2 + (j + 1)
            axis = fig.add_subplot(2, 2, subplot_num)

            for topic in words:
                X = wv[topic]
                X -= mean
                X /= var
                result = pca.transform(X)

                axis.scatter(result[:, 0], result[:, 1], s=5.0)
                for k, word in enumerate(topic):
                    axis.annotate(word, xy=(result[k, 0], result[k, 1]), size=7)

                plt.setp(axis.get_xticklabels(), visible=False)
                plt.setp(axis.get_yticklabels(), visible=False)

            axis.set_title(lang_full.capitalize() + " - " + plot_labels[type] + " - " + plot_labels[hp],
                           fontdict={"fontsize":12})
    # plt.savefig("Figures/embedding_" + lang_short + ".png")

    plt.show()


get_subplot_for_data("nl")
# get_subplot_for_data("en")
# get_subplot_for_data("ru")

