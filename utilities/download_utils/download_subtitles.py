import csv
import os
import requests
from zipfile import ZipFile
import gzip
from utilities.utils import language_map


def download_url(url, DIR):
    filename = url.split("/")[-1]
    save_path = DIR + "/" + filename

    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)


def remove_ids_files():
    main_dir = "data/subtitle_data/"
    for dir in os.listdir(main_dir):
        for file in os.listdir(main_dir + dir):
            if ".ids" in file:
                os.remove(main_dir + dir + "/" + file)
                print("Removed", file)


def download_subtitle_files(datasets):
    print("\nDownloading languages\n")
    for language, url in datasets:
        lang_full, lang_short = language_map(language)
        print("Downloading", language)
        if "raw" in url:
            DIR = "data/subtitle_data/raw"
        else:
            DIR = "data/subtitle_data/en_" + lang_short

        if not os.path.exists(DIR):
            os.mkdir(DIR)

        download_url(url, DIR)


def extract_files():
    print("\nUnzipping files\n")
    main_dir = "data/subtitle_data/"
    for language_dir in os.listdir(main_dir):
        if os.path.isdir(main_dir + language_dir):
            for file in os.listdir(main_dir + language_dir):
                ext = os.path.splitext(file)
                if ext[-1] == ".gz":
                    print("Extracting", file)
                    DIR = main_dir + language_dir
                    new_filename = ext[0]
                    with gzip.GzipFile(DIR + "/" + file, 'rb') as gzip_file:
                        with open(DIR + "/" + new_filename, 'wb') as new_file:
                            for data in iter(lambda: gzip_file.read(1024 * 1024), b''):
                                new_file.write(data)

                    os.remove(DIR + "/" + file)

                if ext[-1] == '.zip':
                    print("Extracting", file)
                    DIR = main_dir + language_dir + "/"
                    with ZipFile(DIR + file, 'r') as f:
                        f.extractall(DIR)

                    os.remove(DIR + file)


def download_all_subtitles(skip_translated=False, skip_raw=True):
    if not os.path.exists("data/subtitle_data"):
        os.mkdir("data/subtitle_data")

    datasets = []

    if not skip_translated:
        print("Downloading translated subtitles")
        with open("utilities/download_utils/subtitle_data.csv", "r") as f:
            new_datatsets = csv.reader(f)
            datasets.extend(list(new_datatsets))

    if not skip_raw:
        print("Downloaded raw subtitles")
        with open("utilities/download_utils/raw_subtitle_data.csv") as f:
            more_datatsets = csv.reader(f)
            datasets.extend(list(more_datatsets))

    download_subtitle_files(datasets)
    extract_files()
    remove_ids_files()

