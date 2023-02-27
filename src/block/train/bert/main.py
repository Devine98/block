# -*- coding: utf-8 -*-
"""
Created on 20220222
@author: songze
"""

import importlib as imp
import pickle

import torch
from models.bert import nets
from trainers.bert import dataset, trainer

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
device = torch.device(device)
n_device = torch.cuda.device_count()
torch.backends.cudnn.is_available()
torch.backends.cudnn.version()
torch.set_default_tensor_type(torch.FloatTensor)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
torch.cuda.set_device(2)


# data 7.5亿
file = open("/data/home/ze.song/data/corpus/game_corpus.bin", "rb")
x = pickle.load(file)
file.close()

file = open("/data/home/ze.song/data/corpus/corpus_wiki.bin", "rb")
y = pickle.load(file)
file.close()

# dataset
imp.reload(dataset)
data_set = dataset.DataSet(x=x, y=y, maxlen=512, maxpred=64)

# model
imp.reload(nets)
model = nets.make_model(vocab_size=50000, maxlen=512, n_layers=8, hidden_size=512)
# model.load_state_dict(torch.load('./model.pkl', map_location="cpu"))

# trainer
imp.reload(trainer)
t = trainer.Trainer(model=model, train_set=data_set)
# t.optim.n_current_steps=148000
t.train(file_path="./check_points/bert/", max_num=2000000)
# t.iteration( t.train_loader,True,"./check_points/bert/",200000)


model.load_state_dict(
    torch.load("./check_points/bert/model_1000000.pkl", map_location="cpu")
)


torch.save(model.bert, "./check_points/bert/bert.pkl")
