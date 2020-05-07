from utilities.download_utils.download_utils import download_url
import bz2
from utilities.utils import language_map
import os
from zipfile import ZipFile


def get_glove_url():
    filename = "glove.6B.zip"
    url = "http://nlp.stanford.edu/data/glove.6B.zip"

    return url, filename


def download_glove():
    print("\nDownloading GloVe files")
    language = "english"

    print("Downloading", language)
    url, filename = get_glove_url()

    files = os.listdir("vector_models/" + language)
    if files:
        already_downloaded = False
        for file in files:
            if "glove" in file:
                print(language, "already downloaded!")
                already_downloaded = True
                break

        if already_downloaded:
            return

    path = "vector_models/" + language + "/" + filename

    download_url(url, path)


def extract_glove_files():
    print("\nExtracting GloVe models\n")
    language_dir = "english/"
    for file in os.listdir("vector_models/" + language_dir):
        ext = os.path.splitext(file)
        if ext[-1] == ".zip" and "glove" in file:
            DIR = "vector_models/" + language_dir
            with ZipFile(DIR + file, 'r') as f:
                f.extractall(DIR)

                os.remove(DIR + file)

            print("Done unzipping", file)