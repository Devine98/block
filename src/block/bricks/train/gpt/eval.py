# -*- coding: utf-8 -*-
"""
Created on 20220222
@author: songze
"""

import importlib as imp
import pickle

import chatbot
import torch

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
device = torch.device(device)
n_device = torch.cuda.device_count()
torch.backends.cudnn.is_available()
torch.backends.cudnn.version()
torch.set_default_tensor_type(torch.FloatTensor)
torch.cuda.set_device(2)

vocab_path = (
    "/data/home/ze.song/git/block/src/block/bricks/train/gpt/model_files/vocab.pkl"
)


def get_latest_model(path="/data/home/ze.song/models/gptb"):

    with open(
        f"{path}/logs.pkl",
        "rb",
    ) as f:
        log = pickle.load(f)

    epoch = log[-1][0].replace("epoch : ", "")
    iter = log[-1][1].replace("iter: ", "")

    model_path = f"{path}/model_{epoch}_{iter}.pkl"
    return model_path


if __name__ == "__main__":

    # large
    imp.reload(chatbot)
    model_path = get_latest_model("/data/home/ze.song/models/gptb")
    print(model_path)
    model_path = "/data/home/ze.song/models/gptb/model_0_790000.pkl"
    model_path = "/data/home/ze.song/models/gpt/model_0_10000.pkl"
    bot3 = chatbot.Bot(model_path=model_path, vocab_path=vocab_path)
    bot3.init()
    bot3.talk("去吃火锅吗?", top_k=None, top_p=0.5)
    bot3.talk("你好", top_k=None, top_p=0.5)
    bot3.talk("再见", top_k=None, top_p=0.6)
    bot3.talk("你是谁", top_k=None, top_p=0.6)
    bot3.talk("在哪里见面", top_k=None, top_p=0.6)
    bot3.talk("你喜欢什么", top_k=None, top_p=0.6)
    bot3.talk("对啊", top_k=None, top_p=0.6)
    bot3.result

    with open(
        "/data/home/ze.song/models/gptb/logs.pkl",
        "rb",
    ) as f:
        a3 = pickle.load(f)
    a3


path = "/data/home/ze.song/data/raw_corpus/wikicorpus/zh/AA/wiki_11"
with open(path, "r") as f:
    a = f.readlines()

    model = torch.nn.Linear(10, 10)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)


def rule(epoch, warmup_steps=10, max_steps=100):
    if max_steps < 10 * warmup_steps:
        max_steps = 10 * warmup_steps
    if epoch < warmup_steps:
        lamda = 5 * epoch / warmup_steps
    elif epoch < 2 * warmup_steps:
        lamda = 5 - 4 * (epoch - warmup_steps) / warmup_steps
    elif epoch < max_steps:
        lamda = 1.3 - (epoch - 2 * warmup_steps) / max_steps
    else:
        lamda = 0.3
    return lamda


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rule)

for i in range(500):
    scheduler.step()

optimizer
