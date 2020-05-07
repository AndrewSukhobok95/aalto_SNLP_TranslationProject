# NLP_vector_project


## Calling the scripts

#### Train

Script location: /TranslationModels/main.py

Example for training:

`python3 main.py -t --type rnn --src en --tgt ru --src_vm ft_en_d100_sg_st --tgt_vm ft_ru_d100_sg_st -m 100000 -b 4 -i 30 -g`

Example for evaluation:

`python3 main.py -e --type rnn --src en --tgt ru --src_vm ft_en_d100_sg_st --tgt_vm ft_ru_d100_sg_st -m 100000 -b 4 -i 30 -g`
`python main.py -e --type rnn --src en --tgt ru --src_vm ft_en_d100_sg_st --tgt_vm ft_ru_d100_sg_st -m 1000`



Description of parameters:

| parameter         | short | default          | possible options   | example            |
| ----------------- | ----- | ---------------- | ------------------ | ------------------ |
| --train           | -t    |                  |                    |                    |
| --eval            | -e    |                  |                    |                    |
| --unfiltered      | -u    |                  |                    |                    |
| --keep_chance     | -k    | 0.9              | 0-1                |                    |
| --type            |       | `tr`             | `rnn` / `tr`       |                    |
| --src             |       | `en`             | `en` / `nl` / `ru` |                    |
| --tgt             |       | `nl`             | `en` / `nl` / `ru` |                    |
| --source_vm       |       |                  |                    | `ft_en_d100_sg_st` |
| --target_vm       |       |                  |                    | `ft_en_d100_sg_st` |
| --hidden_size     |       | 256              |                    |                    |
| --max_batches     | -m    | None (all)       |                    |                    |
| --batch_size      | -b    | 4                |                    |                    |
| --iters           | -i    | 5                |                    |                    |
| --gpu             | -g    |                  |                    |                    |
| --target          |       | I want a dog     |                    |                    |

## Naming convention for the vector models:

All vector models are stored in /data/vector_models/.

The name of any vector model consists of 5 parts.
The are stated in the list below in the order they should be written.
The possible types are given in the brackets.

List of flags for vector model:

- model type (`w2v`, `ft`, `glove`)
- language (`en`, `nl`, `ru`)
- dimension size (`d100`, `d300`)
- type of training (`sg`, `cbow`)
- flag for special tokens added (`st`)
- flag for being downloaded (`d`)

Examples:
- `ft_en_d100_sg_st.bin` - fasttext, english, size=100, skip-gram, special tokens added
- `v2w_nl_d100_cbow.bin` - word2vec, dutch, size=100, CBOW, special tokens are not added
- `ft_en_d100_sg_d.bin` - fasttext, english, size=100, skip-gram, downloaded

Special tokens:
- `<SOS>` - start of sentence
- `<EOS>` - end of sentence
- `<UNK>` - unknown token

## Naming convention for the translation model:

All translation models are stored in /data/translation_models/.

For each model, we will have two saved models: encoder and decoder.
Every saved `.pth` file have the following structure of the name:

| parameter             | possible options       |
| --------------------- | ---------------------- |
| model type            | `rnn` / `tr`           |
| model part            | `encoder` / `decoder`  |
| src language          | `en` / `nl` / `ru`     |
| tgt language          | `en` / `nl` / `ru`     |
| src vector model name |                        |
| tgt vector model name |                        |

Examples:
- `rnn_encoder_en_nl_VM_ft_en_d100_sg_st_VM_v2w_nl_d100_cbow.pth`
- `rnn_decoder_en_nl_VM_ft_en_d100_sg_st_VM_v2w_nl_d100_cbow.pth`

## Naming convention for the data:

All data (training data, vector models, translation models) is located in /data directory.
This directory must have the following structure:

- /data
    - /subtitle_data
        - /en_nl
            - OpenSubtitles.en-nl.en
            - ...
        - /en_ru
            - ...
        - /raw_for_vm
            - ...
    - /train_data
        - /en_nl
            - en_train.txt
            - en_test.txt
            - en_val.txt
            - nl_train.txt
            - nl_test.txt
            - nl_val.txt
        - /en_nl_filtered
            - ...
        - /en_ru
            - ...
        - /en_ru_filtered
            - ...

In order to download the data and filter it:

- downloading without data for training vectors:
    - `python3 download_script.py --skip_raw`
- splitting the downloaded data in train/test/val:
    - `python3 utilities/split_subtitles.py --filter`



