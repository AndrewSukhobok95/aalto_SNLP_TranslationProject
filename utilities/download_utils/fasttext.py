import gzip
from utilities.download_utils.download_utils import download_url
from utilities.utils import language_map
import os


def get_fasttext_url(language):
    _, language = language_map(language)

    filename = "cc." + language + ".300.vec.gz"
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/" + filename

    return url, filename


def download_fasttext(languages):
    print("\nDownloading FastText files\n")
    for language in languages:
        print("Downloading", language)
        url, filename = get_fasttext_url(language)
        print(url)

        files = os.listdir("vector_models/" + language)
        if files:
            already_downloaded = False
            for file in files:

                if ".vec" in file:
                    print(language, "already downloaded!")
                    already_downloaded = True
                    break

            if already_downloaded:
                continue

        path = "vector_models/" + language + "/" + filename

        download_url(url, path)


def extract_fasttext_files():
    print("\nExtracting FastText models\n")
    for language_dir in os.listdir("vector_models"):
        print("Extracting", language_dir)
        for file in os.listdir("vector_models/" + language_dir):
            ext = os.path.splitext(file)
            if ext[-1] == ".gz":
                DIR = "vector_models/" + language_dir
                new_filename = ext[0]
                with gzip.GzipFile(DIR + "/" + file, 'rb') as gzip_file:
                    with open(DIR + "/" + new_filename, 'wb') as new_file:
                        for data in iter(lambda : gzip_file.read(1024 * 1024), b''):
                            new_file.write(data)

                os.remove(DIR + "/" + file)