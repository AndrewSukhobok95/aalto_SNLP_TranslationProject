

with open("vector_models/dutch/nlwiki_20180420_300d.txt.bz2", "rb") as f:
    for line in f.readlines()[:30]:
        print(line)