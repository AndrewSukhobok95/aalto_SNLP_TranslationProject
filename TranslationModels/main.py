import os
import os.path
import numpy as np
import gensim
from gensim.test.utils import datapath as gensim_datapath
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import FastTextKeyedVectors

import sys
sys.path.append('./../')
sys.path.append(os.path.abspath(os.path.normpath(os.path.join(__file__, "./../../"))))
import argparse

from TranslationModels.const_vars import *
from TranslationModels.rnn_model import RNNModel
from TranslationModels.dataloader import tr_data_loader
from TranslationModels.transformer_model import TransformerModel

import matplotlib.pyplot as plt

def extendPretrainedModel(model):
    length = model.vector_size
    try:
        model.get_vector(SOS_token)
    except:
        model.add(SOS_token, np.random.normal(0, 0.01, length))
    try:
        model.get_vector(EOS_token)
    except:
        model.add(EOS_token, np.random.normal(0, 0.01, length))
    try:
       model.get_vector(UNK_token)
    except:
        model.add(UNK_token, np.random.normal(0, 0.01, length))
    return model

def read_vector_models(path_src_vw_model_bin, path_tgt_vw_model_bin):
    if not all([os.path.isfile(fname) for fname in [path_src_vw_model_bin, path_tgt_vw_model_bin]]):
        print('Some of the vector model files given do not exist, perhaps check defaults!')
        sys.exit()

    print('+ preparing src vector model')
    if "ft" in path_src_vw_model_bin:
        vw_src_model = FastTextKeyedVectors.load(path_src_vw_model_bin)
        vw_src_model.add(UNK_token, np.random.normal(0, 0.01, vw_src_model.vector_size))
    else:
        vw_src_model = KeyedVectors.load_word2vec_format(path_src_vw_model_bin, binary=True)
    print('++ src vector model read')
    vw_src_model = extendPretrainedModel(vw_src_model)
    print('++ src vector model extended')

    print('+ preparing tgt vector model')
    if "ft" in path_tgt_vw_model_bin:
        vw_tgt_model = FastTextKeyedVectors.load(path_tgt_vw_model_bin)
        vw_tgt_model.add(UNK_token, np.random.normal(0, 0.01, vw_tgt_model.vector_size))
    else:
        vw_tgt_model = KeyedVectors.load_word2vec_format(path_tgt_vw_model_bin, binary=True)
    print('++ tgt vector model read')
    vw_tgt_model = extendPretrainedModel(vw_tgt_model)
    print('++ tgt vector model extended')

    return vw_src_model, vw_tgt_model


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', action = 'store_true', help='Should training be performed.')
    parser.add_argument('--eval', '-e', action = 'store_true', help='Should evaluation be performed.')
    parser.add_argument('--type', type = str, choices = ['rnn', 'tr'], help='Type of translation model.', default = 'tr')

    parser.add_argument('--src', choices = ['en', 'nl', 'ru'], help='Source language for translation.', default = 'en')
    parser.add_argument('--tgt', choices = ['en', 'nl', 'ru'], help='Target language for translation.', default = 'nl')

    parser.add_argument('--source_vm', type = str, help='Word vectors for the source language, filename in the data/vector_models folder.', required = True)
    parser.add_argument('--target_vm', type = str, help='Paired corpus in the target language, filename in the data/vector_models folder.', required = True)


    parser.add_argument('--hidden_size', type = int, help='', default = 256)
    parser.add_argument('--keep_chance', '-k', type = float, help='', default = 0.9)
    parser.add_argument('--max_batches', '-m', type = int, help='Maximum number of batches.', default = None)
    parser.add_argument('--batch_size', '-b', type = int, help='Batch size.', default = 4)
    parser.add_argument('--iters', '-i', type = int, help='Number of iterations.', default = 5)
    parser.add_argument('--gpu', '-g', action = 'store_true', help='Should training be done on GPU.')
    parser.add_argument('--unfiltered', '-u', action = 'store_const', help='Use unfiltered data.', const = '', default = '_filtered')

    parser.add_argument('--target', action = 'extend', type = str, help='Sentence to translate.', default = ['I want a dog'])

    args = parser.parse_args()

    if args.src == args.tgt:
        print('Source and target language identical!')
        sys.exit()

    path_src_vw_model_bin = './../data/vector_models/' + args.source_vm + '.bin'
    path_tgt_vw_model_bin = './../data/vector_models/' + args.target_vm + '.bin'

    vw_src_model, vw_tgt_model = read_vector_models(path_src_vw_model_bin, path_tgt_vw_model_bin)

    translation_models_path = './../data/translation_models/'
    if not os.path.exists(translation_models_path):
        os.makedirs(translation_models_path)

    enc_path = translation_models_path + args.type + '_encoder_' + args.src + '_' + args.tgt + '_VM_' + args.source_vm + '_VM_' + args.target_vm + '.pth'
    dec_path = translation_models_path + args.type + '_decoder_' + args.src + '_' + args.tgt + '_VM_' + args.source_vm + '_VM_' + args.target_vm + '.pth'

    if args.type == 'rnn':
        translation_model = RNNModel(
             src_vectorModel=vw_src_model,
             tgt_vectorModel=vw_tgt_model,
             encoder_save_path=enc_path,
             decoder_save_path=dec_path,
             hidden_size=args.hidden_size)
    else:
        translation_model = TransformerModel(
             src_vectorModel=vw_src_model,
             tgt_vectorModel=vw_tgt_model,
             encoder_save_path=enc_path,
             decoder_save_path=dec_path,
             hidden_size=args.hidden_size)

    if args.train:

        path_src_train_file = './../data/train_data/' + min(args.src, args.tgt) + '_' + max(args.src, args.tgt) + args.unfiltered + '/' + args.src + '_train.txt' 
        path_tgt_train_file = './../data/train_data/' + min(args.src, args.tgt) + '_' + max(args.src, args.tgt) + args.unfiltered + '/' + args.tgt + '_train.txt' 
        
        if not all([os.path.isfile(fname) for fname in [path_src_train_file, path_tgt_train_file]]):
            print('Some of the train files given do not exist, perhaps check defaults!')
            sys.exit()

        print('+ start TrNN training')
        translation_model.train(
             path_src_train_file,
             path_tgt_train_file,
             batch_size=args.batch_size,
             iters=args.iters,
             max_batches = args.max_batches,
             device = 'cuda:0' if args.gpu else 'cpu',
             keep_chance=args.keep_chance
             )

    if args.eval:
        path_src_test_file = './../data/train_data/' + min(args.src, args.tgt) + '_' + max(args.src, args.tgt) + args.unfiltered + '/' + args.src + '_test.txt' 
        path_tgt_test_file = './../data/train_data/' + min(args.src, args.tgt) + '_' + max(args.src, args.tgt) + args.unfiltered + '/' + args.tgt + '_test.txt'

        eval_file = './../data/eval_results/' + args.type + "_" + args.src + '_' + args.tgt + '_VM_' + args.source_vm + '_VM_' + args.target_vm + '.txt'
        if not os.path.exists("./../data/eval_results/"):
            os.mkdir("./../data/eval_results/")

        if not all([os.path.isfile(fname) for fname in [path_src_test_file, path_tgt_test_file]]):
            print('Some of the test files given do not exist, perhaps check defaults!')
            sys.exit()

        if args.max_batches > 5000:
            args.max_batches = 5000

        print('+ start evaluation')
        scores = translation_model.eval(
             path_src_test_file,
             path_tgt_test_file,
             eval_file,
             batch_size=1,
             max_batches = args.max_batches,
             keep_chance=args.keep_chance,
             device = 'cuda:0' if args.gpu else 'cpu',
             )
        plt.hist(scores)
        plt.savefig(eval_file[:-3] + 'png')
        print('+ Evaluation done')

    sample_sentences = [
        "I want to buy a cat.",
        "I want him to be a doctor.",
        "I wish this trip would never end.",
        "Whoever sells the most wins a trip to Disneyland.",
        "Jordanian law does not impose any gender-based conditions upon passport applicants.",
        "Let's consider this thought as an offer.",
        "Do you want to talk about it?",
        "How long have you been in Paris?"
    ]

    print()
    print("Translation of sample sentences:")
    for s in sample_sentences:
        tr_res = translation_model.translate(s, True)
        print('++ Input:', s)
        print('++ Output:', " ".join(tr_res[:-1]))
        print("================================================")

    print()
    for tr_input in args.target:
        tr_res = translation_model.translate(tr_input, True)
        print('+ Example translation:')
        print('++ Input:', tr_input)
        print('++ Output:', tr_res)
        print('\n\n')

    print()
    print('done!')
