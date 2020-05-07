from train_embeddings import train_model
import argparse
import logging
import os
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loops", type=int, help="Number of full loops over the corpus", default=10)
    parser.add_argument("--chunks", type=int, help="Number of chunks to split corpus in", default=10)
    parser.add_argument("--epochs", type=int, help="Number of epochs per chunk", default=5)
    parser.add_argument("--dim", type=int, help="Embedding dimension", default=100)
    parser.add_argument("--log", type=bool, help="Pass if you want gensim to print logs")
    parser.add_argument("--continue_training", type=bool, default=True)
    parser.add_argument("--special_tokens", type=bool, default=True)
    parser.add_argument("--fasttext", action='store_true')
    parser.add_argument("--cores", type=int, default=6)
    parser.add_argument("--superloops", type=int, default=100)
    parser.add_argument("--skip_gram", action='store_true')
    args = parser.parse_args()

    args.loops = 1
    args.chunks = 10
    args.epochs = 1

    if args.log:
        print("Gensim will output logs!")
        logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

    if args.fasttext:
        print("Training FastText models")
    else:
        print("Training Word2Vec models")

    if not os.path.exists("trained_models"):
        os.mkdir("trained_models")

    if args.special_tokens:
        print("Adding tokens <SOS> and <EOS>")

    languages = ["dutch", "english", "russian"]

    for superloop in range(args.superloops):
        print("Superloop", superloop)
        random.shuffle(languages)

        for lang in languages:
            args.epochs = int(max(5 - superloop, 1))
            args.language = lang
            if args.language != "english":
                args.loops = 3
            else:
                args.loops = 1

            print("Language", args.language)
            train_model(args)