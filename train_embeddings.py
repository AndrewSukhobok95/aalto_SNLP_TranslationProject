from gensim.models import Word2Vec, FastText
from utilities.utils import language_map, preprocess, load_subtitles,\
    get_num_lines, add_special_tokens, load_random_subtitles
import logging
import argparse
import random
import math
import multiprocessing
import os


def get_model_path(args):
    lang_full, lang_short = language_map(args.language)

    if args.fasttext:
        model_name = "ft_"
    else:
        model_name = "w2v_"

    model_name += lang_short + "_d" + str(args.dim)

    if args.skip_gram:
        model_name += "_sg"
    else:
        model_name += "_cbow"

    if args.special_tokens:
        model_name += "_st"

    model_name += ".model"
    model_path = "trained_models/" + lang_full + "/" + model_name

    print("Model name", model_name)
    return model_name, model_path


def set_sample_prob(args):
    _, lang_short = language_map(args.language)

    filename = "OpenSubtitles.raw." + lang_short
    file = "data/subtitle_data/raw/" + filename

    num_lines = get_num_lines(file)
    print("Total number of lines:", num_lines)
    chunk_size = int(num_lines / args.chunks)
    if chunk_size > 5e6:
        chunk_size = 5e6
        args.chunks = int(math.ceil(num_lines / chunk_size))
        print("Chunk size too large, set to", int(chunk_size), "with", args.chunks, "chunks!")
    p = chunk_size / num_lines
    return p


def get_model(args, model_path):
    cores = max(1, multiprocessing.cpu_count() - 2)
    if os.path.exists(model_path) and args.continue_training:
        print("Continuing training existing model")
        if args.fasttext:
            model = FastText.load(model_path)
        else:
            model = Word2Vec.load(model_path)
    else:
        if args.fasttext:
            model = FastText(min_count=20,
                             window=5,
                             size=args.dim,
                             alpha=0.03,
                             min_alpha=0.0007,
                             workers=cores,
                             sample=6e-5,
                             negative=20,
                             sg=args.skip_gram)
        else:
            model = Word2Vec(min_count=20,
                             window=5,
                             size=args.dim,
                             alpha=0.03,
                             min_alpha=0.0007,
                             workers=cores,
                             sample=6e-5,
                             negative=20,
                             sg=args.skip_gram)
    if args.cores == -1:
        model.workers = cores
    else:
        model.workers = args.cores

    return model


def train_chunk(model, language, p, epochs, special_tokens):
    subtitles = load_random_subtitles(language, p)
    preprocess(subtitles, False)
    if special_tokens:
        add_special_tokens(subtitles)
        assert "<SOS>" in model
        assert "<EOS>" in model

    model.build_vocab(subtitles, progress_per=10000, update=model.wv.vocab)
    model.train(subtitles, total_examples=len(subtitles), epochs=epochs)


def train_model(args):
    lang_full, lang_short = language_map(args.language)
    model_name, model_path = get_model_path(args)

    print("Training on", lang_full, "- Embedding size:", args.dim, "- Loops:", args.loops,
          "- Chunks:", args.chunks, "- Epochs:", args.epochs)

    if args.skip_gram:
        print("Using skip gram")
    else:
        print("Using CBOW")

    model = get_model(args, model_path)
    sample_p = set_sample_prob(args)

    for loop in range(args.loops):
        epochs = int(max(1, args.epochs - loop))
        chunk_list = list(range(args.chunks))
        for i, chunk in enumerate(chunk_list):
            print("Loop", loop + 1, "/", args.loops, "- Chunk, ", i + 1, "/", args.chunks, "- Epochs:", epochs)

            train_chunk(model, args.language, sample_p, epochs, args.special_tokens)
            model.save(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, help="Language to train", default="en")
    parser.add_argument("--loops", type=int, help="Number of full loops over the corpus", default=10)
    parser.add_argument("--chunks", type=int, help="Number of chunks to split corpus in", default=10)
    parser.add_argument("--epochs", type=int, help="Number of epochs per chunk", default=5)
    parser.add_argument("--dim", type=int, help="Embedding dimension", default=100)
    parser.add_argument("--log", type=bool, help="Pass if you want gensim to print logs")
    parser.add_argument("--continue_training", type=bool, default=True)
    parser.add_argument("--special_tokens", type=bool, default=True)
    parser.add_argument("--skip_gram", action='store_true')
    parser.add_argument("--fasttext", action='store_true')
    parser.add_argument("--cores", type=int, default=-1)
    args = parser.parse_args()

    if args.log:
        print("Gensim will output logs!")
        logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

    if args.fasttext:
        print("Training FastText model")
    else:
        print("Training Word2Vec model")

    if not os.path.exists("trained_models"):
        os.mkdir("trained_models")

    lang, _ = language_map(args.language)

    if not os.path.exists("trained_models/" + lang):
        os.mkdir("trained_models/" + lang)

    if args.special_tokens:
        print("Adding tokens <SOS> and <EOS>")

    train_model(args)





