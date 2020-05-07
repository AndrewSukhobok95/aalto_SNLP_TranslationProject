from utilities.download_utils.download_utils import download_url
import bz2
from utilities.utils import language_map
import os


def get_wikipedia2vec_url(language, dim=300):
    _, language = language_map(language)

    filename = language + "wiki_20180420_" + str(dim) + "d.txt.bz2"
    url = "http://wikipedia2vec.s3.amazonaws.com/models/" + language + "/2018-04-20/" + filename

    return url, filename


def download_wikipedia2vec(languages, dim=300):
    print("\nDownloading wikipedia2vec files of dimension", dim, "\n")
    for language in languages:
        print("Downloading", language)
        url, filename = get_wikipedia2vec_url(language, dim)

        files = os.listdir("vector_models/" + language)
        if files:
            already_downloaded = False
            for file in files:
                if str(dim) in file:
                    print(language, "already downloaded for dim=", dim)
                    already_downloaded = True
                    break

            if already_downloaded:
                continue

        path = "vector_models/" + language + "/" + filename

        download_url(url, path)


def extract_wikipedia2vec_files():
    print("\nExtracting wikipedia2vec models\n")
    for language_dir in os.listdir("vector_models"):
        print("Extracting", language_dir)
        for file in os.listdir("vector_models/" + language_dir):
            ext = os.path.splitext(file)
            if ext[-1] == ".bz2":
                DIR = "vector_models/" + language_dir
                new_filename = ext[0]
                with bz2.BZ2File(DIR + "/" + file, 'rb') as bz2_file:
                    with open(DIR + "/" + new_filename, 'wb') as new_file:
                        for data in iter(lambda : bz2_file.read(1024 * 1024), b''):
                            new_file.write(data)

                os.remove(DIR + "/" + file)