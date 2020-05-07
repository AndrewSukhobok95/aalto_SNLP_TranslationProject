from utilities.download_utils.download_subtitles import download_all_subtitles
from utilities.download_utils.download_vector_models import download_all_vector_models
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_raw", action="store_true")
    parser.add_argument("--skip_translations", action="store_true")

    args = parser.parse_args()

    download_all_subtitles(args.skip_translations, args.skip_raw)