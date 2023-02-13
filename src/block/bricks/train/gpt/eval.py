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
torch.cuda.set_device(0)

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

    # small
    imp.reload(chatbot)
    model_path = get_latest_model("/data/home/ze.song/models/gpt_")
    bot = chatbot.Bot(model_path=model_path, vocab_path=vocab_path)
    bot.init()
    bot.talk("去吃火锅吗?")
    bot.talk("你好")
    bot.talk("再见")
    bot.result

    with open(
        "/data/home/ze.song/models/gpt_/logs.pkl",
        "rb",
    ) as f:
        a = pickle.load(f)
    a

    # base
    imp.reload(chatbot)
    with open(
        "./model_files/config.pkl",
        "rb",
    ) as f:
        config = pickle.load(f)
    model_path = get_latest_model("/data/home/ze.song/models/gpt")
    bot2 = chatbot.Bot(model_path=model_path, vocab_path=vocab_path, config=config)
    bot2.init()
    bot2.talk("去吃火锅吗?", top_k=None, top_p=0.7)
    bot2.talk("你好")
    bot2.talk("再见")
    bot2.talk("你是谁")
    bot2.talk("在哪里见面")
    bot2.talk("你喜欢什么")
    bot2.talk("对啊")
    bot2.result

    with open(
        "/data/home/ze.song/models/gpt/logs.pkl",
        "rb",
    ) as f:
        a2 = pickle.load(f)
    a2

    # large
    imp.reload(chatbot)
    config = {
        "vocab_size": 13317,
        "embd_pdrop": 0.1,
        "n_embd": 1536,
        "n_head": 24,
        "n_positions": 320,
        "n_layer": 18,
        "attn_pdrop": 0.1,
        "resid_dropout": 0.1,
        "n_inner": 1536 * 4,
        "layer_norm_epsilon": 1e-5,
        "pad_idx": 0,
        "dtype": torch.float32,
        "segment_size": 3,
    }
    model_path = get_latest_model("/data/home/ze.song/models/gptb")
    bot3 = chatbot.Bot(model_path=model_path, vocab_path=vocab_path, config=config)
    bot3.init()
    bot3.talk("去吃火锅吗?", top_k=None, top_p=0.7)
    bot3.talk("你好", top_k=None, top_p=0.8)
    bot3.talk("再见", top_k=None, top_p=0.7)
    bot3.talk("你是谁")
    bot3.talk("在哪里见面")
    bot3.talk("你喜欢什么")
    bot3.talk("对啊")
    bot3.result

    with open(
        "/data/home/ze.song/models/gptb/logs.pkl",
        "rb",
    ) as f:
        a3 = pickle.load(f)
    a3
