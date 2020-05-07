from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_multiple_whitespaces, strip_punctuation
from nltk.tokenize import WordPunctTokenizer
import random


def add_special_tokens(lines):
    for i, line in enumerate(lines):
        line.insert(0, "<SOS>")
        line.append("<EOS>")
        lines[i] = line


def preprocess(lines, remove_punctuation=True):
    for i, line in enumerate(lines):
        lines[i] = preprocess_line(line, remove_punctuation)


def preprocess_line(line, remove_punctuation=True):
    if remove_punctuation:
        return preprocess_string(line, [lambda x: x.lower(), strip_tags, strip_multiple_whitespaces, strip_punctuation])
    else:
        return WordPunctTokenizer().tokenize(line.lower())


def language_map(lang):
    lang = lang.lower()

    if lang == "dutch" or lang == "nl":
        return ["dutch", "nl"]

    if lang == "russian" or lang == "ru":
        return ["russian", "ru"]

    if lang == "english" or lang == "en":
        return ["english", "en"]

    print("Warning: Language not detected, returning None")
    return None, None


def load_random_subtitles(lang, p):
    lang_full, lang_short = language_map(lang)

    filename = "OpenSubtitles.raw." + lang_short
    file = "data/subtitle_data/raw/" + filename

    with open(file) as f:
        subs = []
        line = f.readline()
        while line:
            if random.random() < p:
                subs.append(line)

            line = f.readline()

        return subs


def load_subtitles(lang="nl", size=-1, start=None, end=None):
    lang_full, lang_short = language_map(lang)

    filename = "OpenSubtitles.raw." + lang_short

    file = "data/subtitle_data/raw/" + filename

    with open(file) as f:
        if start is not None and end is not None:
            subs = []
            line = f.readline()
            ind = 0
            while line:
                if start <= ind <= end:
                    subs.append(line)

                line = f.readline()
                ind += 1
                if ind > end:
                    break

            return subs

        elif size == -1:
            subs = f.read().splitlines()
            return subs
        else:
            subs = []
            for i in range(size):
                sentence = f.readline()
                subs.append(sentence)

            return subs


def get_num_lines(path):
    with open(path) as f:
        line = f.readline()
        lines = 0
        while line:
            line = f.readline()
            lines += 1

        return lines


def keep_lines(line1, line2, threshold=0.55):
    if len(line1) > 500 or len(line2) > 500:
        return False

    avg_length = (len(line1) + len(line2)) / 2

    threshold_length = max(12, threshold * avg_length)

    diff = abs(len(line1) - len(line2))
    return diff <= threshold_length
