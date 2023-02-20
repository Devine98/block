import json
import os
import pickle

import zhconv
from tqdm import tqdm


def save_pkl(path, file):
    if os.path.exists(path):
        os.remove(path)
    with open(path, "wb") as f:
        pickle.dump(file, f)


def load_pkl(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


def process_50w(path):
    """process 50w dialogue"""
    with open(path, "r") as f:
        d = f.readlines()

    corpus = []
    res = []
    for x in tqdm(d):
        x = x.strip()
        if x == "":
            corpus.append(res)
            res = []
        else:
            res.append(x)
    return corpus


def process_lccc(path):
    with open(path, "r") as f:
        b = json.load(f)

    corpus = []
    for s in tqdm(b):
        res = []
        for t in s:
            t = t.strip()
            this_res = ""
            for i, ch in enumerate(t):
                if ch == " ":
                    if ord(t[i - 1]) <= 128 and ord(t[i + 1]) <= 128:
                        this_res = this_res + ch
                else:
                    this_res = this_res + ch
            res.append(this_res)
        corpus.append(res)
    return corpus


def wiki_process(path):
    corpus = []
    doc = []
    for root, _, names in os.walk(path):
        for name in tqdm(names):
            file_path = f"{root}/{name}"
            with open(file_path, "r") as f:
                a = f.readlines()
                a = [i for i in a if i != "\n"]
                a = [zhconv.convert(i, "zh-hans") for i in a]
            for s in a:
                if s.startswith("<doc id="):
                    doc = []
                elif s.startswith("</doc>"):
                    corpus.append(doc)
                    doc = []
                else:
                    doc.append(s.strip())
    return corpus


if __name__ == "__main__":

    # 50w
    path = "/data/home/ze.song/data/raw_corpus/dialogue/50w/train.txt"
    corpus = process_50w(path)
    path = "/data/home/ze.song/data/corpus/dialogue/50w.pkl"
    save_pkl(path, corpus)

    # 100w
    path = "/data/home/ze.song/data/raw_corpus/dialogue/100w/train_100w.txt"
    corpus = process_50w(path)
    path = "/data/home/ze.song/data/corpus/dialogue/100w.pkl"
    save_pkl(path, corpus)

    # LCCC
    path = "/data/home/ze.song/data/raw_corpus/dialogue/LCCD.json"
    corpus = process_lccc(path)
    path = "/data/home/ze.song/data/corpus/dialogue/lccc_large.pkl"
    save_pkl(path, corpus)

    path = "/data/home/ze.song/data/raw_corpus/dialogue/LCCC-base_train.json"
    corpus = process_lccc(path)
    path = "/data/home/ze.song/data/corpus/dialogue/lccc_base.pkl"
    save_pkl(path, corpus)

    # merge
    path = "/data/home/ze.song/data/corpus/dialogue/50w.pkl"
    c1 = load_pkl(path)
    path = "/data/home/ze.song/data/corpus/dialogue/100w.pkl"
    c2 = load_pkl(path)
    path = "/data/home/ze.song/data/corpus/dialogue/lccc_large.pkl"
    c3 = load_pkl(path)
    path = "/data/home/ze.song/data/corpus/dialogue/lccc_base.pkl"
    c4 = load_pkl(path)
    corpus = c1 + c2 + c3 + c4
    path = "/data/home/ze.song/data/corpus/dialogue/corpus.pkl"
    save_pkl(path, corpus)

    # wiki
    path = "/data/home/ze.song/data/raw_corpus/wikicorpus/zh/AA"
    corpus = wiki_process(path)
    path = "/data/home/ze.song/data/corpus/zh_wiki.pkl"
    save_pkl(path, corpus)
