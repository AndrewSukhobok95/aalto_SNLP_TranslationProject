import random
import matplotlib.pyplot as plt
from utilities.utils import get_num_lines, keep_lines, language_map
import numpy as np
import pickle as p


def sample_lines(num_lines, threshold=0.55, only_kept=False, only_discarded=False, lang="nl"):
    _, lang_short = language_map(lang)

    path_en = "data/subtitle_data/en_" + lang_short + "/OpenSubtitles.en-" + lang_short + ".en"
    path_to = "data/subtitle_data/en_" + lang_short + "/OpenSubtitles.en-" + lang_short + "." + lang_short

    kept = 0
    discarded = 0

    with open(path_en, "r") as file1, open(path_to) as file2:
        line1 = file1.readline()
        line2 = file2.readline()
        while line1 and line2 and (kept < num_lines or discarded < num_lines):
            if random.random() < 0.0001:
                diff = abs(len(line1) - len(line2)) / np.mean((len(line1), len(line2)))
                if keep_lines(line1, line2, threshold):
                    if kept < num_lines and diff > threshold - 0.1:
                        kept += 1
                        if not only_discarded:
                            print("\nKEEPING")
                            print("Diff", diff)
                            print(line1)
                            print(line2)
                elif discarded < num_lines:
                    discarded += 1
                    if not only_kept:
                        print("\nDISCARDING")
                        print("Diff", diff)
                        print(line1)
                        print(line2)

            line1 = file1.readline()
            line2 = file2.readline()


def line_shrinkage(thresholds, lang="nl"):
    _, lang_short = language_map(lang)

    path_en = "data/subtitle_data/en_" + lang_short + "/OpenSubtitles.en-" + lang_short + ".en"
    path_to = "data/subtitle_data/en_" + lang_short + "/OpenSubtitles.en-" + lang_short + "." + lang_short

    p = 0.1

    lines_sampled = 0
    num_lines = get_num_lines(path_en)

    original_lengths = 0

    new_lengths = np.zeros(len(thresholds))
    lines_kept = np.zeros(len(thresholds))

    counter = 0
    with open(path_en, "r") as file1, open(path_to) as file2:
        line1 = file1.readline()
        line2 = file2.readline()
        while line1 and line2:
            counter += 1
            if counter % 1e7 == 0:
                print("Read", counter, "out of", num_lines)

            if random.random() < p:
                lines_sampled += 1

                apr_length = (len(line1) + len(line2)) / 2
                for i, threshold in enumerate(thresholds):
                    if keep_lines(line1, line2, threshold):
                        lines_kept[i] += 1
                        new_lengths[i] += apr_length

                original_lengths += apr_length

            line1 = file1.readline()
            line2 = file2.readline()

    new_lengths /= lines_kept
    original_lengths /= lines_sampled

    return new_lengths / original_lengths


def get_percentage(thresholds, lang="nl"):
    _, lang_short = language_map(lang)

    path_en = "data/subtitle_data/en_" + lang_short + "/OpenSubtitles.en-" + lang_short + ".en"
    path_to = "data/subtitle_data/en_" + lang_short + "/OpenSubtitles.en-" + lang_short + "." + lang_short

    sample = 0.1

    num_lines = get_num_lines(path_en)
    lines_sampled = 0

    counts = np.zeros(len(thresholds))
    counter = 0
    with open(path_en, "r") as file1, open(path_to) as file2:
        line1 = file1.readline()
        line2 = file2.readline()
        while line1 and line2:
            counter += 1
            if counter % 1e7 == 0:
                print("Read", counter, "out of", num_lines)

            if random.random() < sample:
                lines_sampled += 1
                for i, threshold in enumerate(thresholds):
                    if keep_lines(line1, line2, threshold):
                        counts[i] += 1
                    else:
                        break

            line1 = file1.readline()
            line2 = file2.readline()

    return counts / lines_sampled


def plot_cutoff(search_again=False):
    threshold_range = np.linspace(0.9, 0.4, 20)

    if search_again:
        print("Calculating average shrinkages")
        perc_length = line_shrinkage(threshold_range)
        print("Calculating average percentages kept")
        perc_kept = get_percentage(threshold_range)

        with open("lengths.p", "wb") as f:
            p.dump(perc_length, f)

        with open("kept.p", "wb") as f:
            p.dump(perc_kept, f)

    else:
        with open("lengths.p", "rb") as f:
            perc_length = p.load(f)

        with open("kept.p", "rb") as f:
            perc_kept = p.load(f)

    min_diff = 100
    best_threshold = 0
    for threshold, length_percentage, percentage_kept in zip(threshold_range, perc_length, perc_kept):
        print(threshold, percentage_kept)
        diff = abs(length_percentage - 1.0)
        if diff < min_diff:
            best_threshold = threshold
            min_diff = diff

    print("Threshold closest to 1.0:", best_threshold)

    plt.plot(threshold_range, perc_kept)
    plt.plot(threshold_range, perc_length)
    # plt.vlines(best_threshold, min(perc_kept), max(perc_length))
    plt.legend(["Data kept (%)", "Average line length (%)"])
    plt.xlabel("Cut-off threshold (%)")
    plt.ylabel("%")
    plt.show()


# sample_lines(20, only_discarded=False)
plot_cutoff(False)