import collections
import os
import pickle
import warnings
from typing import Dict, List, Set, Tuple

import hdfs
import typer

from block import MLPCorrector
from block.global_conf import HDFS_ADDRESS, HDFS_PATH
from block.spelling_correction.trainers.mlp.trainer import Trainer

app = typer.Typer()


def hdfs_download(path, hdfs_path) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    client = hdfs.InsecureClient(HDFS_ADDRESS)
    client.download(
        hdfs_path=hdfs_path,
        local_path=path,
        overwrite=True,
    )


def read_conll(file_path: str) -> Tuple:
    """Reads conll training data,including raw words and misssp's correct form."""
    words, labels = [], []
    with open(file_path) as f:
        word, pos = [], []
        for line in f.readlines():
            sp = line.strip("\n").split("\t")
            if sp == [""]:
                words.append(word)
                labels.append(pos)
                word = []
                pos = []
            else:
                word.append(sp[0])
                if len(sp) > 1:
                    pos.append(sp[1])
                else:
                    pos.append("<correct>")
    return words, labels


def load_never_correct(path) -> Set[str]:
    """Loads pre organized free fire words."""
    dictionary = []
    with open(path) as f:
        for line in f.readlines():
            s = line.strip("\n")
            dictionary.append(s)
            dictionary.append(s.lower())
    return set(dictionary)


def load_missp(path) -> Dict[str, List[str]]:
    """Loads pre organized  misssps and their correct form."""
    c_words = []
    m_words = []
    with open(path) as f:
        words = []
        for line in f.readlines():
            s = line.strip("\n")
            if s.startswith("$"):
                if words != []:
                    m_words.append(words)
                    words = []
                c_words.append(s.replace("$", "").lower())
            else:
                words.append(s.lower())
        m_words.append(words)
    m2c = collections.defaultdict(lambda: [])
    for i in range(len(m_words)):
        c = c_words[i]
        for w in m_words[i]:
            if w == c:
                continue
            if m2c[w] == []:
                m2c[w] = [c]
            else:
                if c in m2c[w]:
                    continue
                else:
                    m2c[w].append(c)
    return dict(m2c)


def get_training_data(local_path: str) -> None:
    """Downloading all training data for mlp corrector."""
    for file in [
        "game_with_random_4m_wiki.bin",
        "missp.txt",
        "mlp_corrector_train_data.txt",
        "never_correct.txt",
    ]:
        hdfs_download(
            path=local_path, hdfs_path=f"{HDFS_PATH}/spelling_correction/{file}"
        )


@app.command()
def main(
    local_path: str = "./train_data_for_mlp_corrector",
    save_path: str = "./mlp_corrector_model.pkl",
    batch_size: int = 8,
    max_step: int = 2000,
    num_workers: int = 1,
) -> None:
    """Traing mlp corrector.



    Args:
        local_path (str, optional):
            Local path for downloading data.
            Defaults to ".".
        save_path (str, optional):
            Local model path.
            Defaults to "./model.pkl".
        batch_size (int, optional):
            Batch size for training rank model.
            Defaults to 8.
        max_step (int, optional):
            Max step for training rank model.
            Defaults to 2000.
        num_workers (int, optional):
            Num workers for counting and training rank model.
            Defaults to 1.
    """

    warn_msg = """
        This code uses data processed by ze.song.
        Please check the data carefully before running this code!
        Do not use this code if you want to collect your own training data!
        """
    warnings.warn(warn_msg)

    # get data
    print("downloading data...")
    get_training_data(local_path)

    print("loading data...")
    with open("/data/home/ze.song/data/corpus/game_with_random_4m_wiki.bin", "rb") as f:
        corpus = pickle.load(f)

    never_correct = load_never_correct(f"{local_path}/never_correct.txt")
    m2c = load_missp(f"{local_path}/missp.txt")
    words, labels = read_conll(f"{local_path}/mlp_corrector_train_data.txt")

    model = MLPCorrector()
    trainer = Trainer(model)

    # training
    trainer.add_never_correct(never_correct)
    trainer.add_m2c(m2c)

    print("training...")
    trainer.train(
        corpus=corpus,
        words=words,
        labels=labels,
        correct_token="<correct>",
        batch_size=batch_size,
        max_step=max_step,
        num_workers=num_workers,
    )

    print("saving...")
    trainer.save_model(path=save_path)


if __name__ == "__main__":
    app()
