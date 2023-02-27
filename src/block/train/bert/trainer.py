import collections
import os
import pickle

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from . import optim


def save_file(f, path):
    if os.path.exists(path):
        os.remove(path)
    file = open(path, "wb")
    pickle.dump(f, file)
    file.close()


class Trainer:
    """
    dan任务
        1. MLM
    """

    def __init__(
        self,
        train_set,
        model,
        batch_size=64,
        val_set=None,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
        warmup_steps=2000,
        cuda: bool = True,
        cuda_devices=None,
        log_freq: int = 5000,
        save_freq: int = 20000,
        opt_freq=4,
    ):
        """
        :param model: NET
        :param vocab_size: meaningful word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.betas = betas
        self.weight_decay = weight_decay
        cuda = torch.cuda.is_available() and cuda
        self.device = torch.device("cuda" if cuda else "cpu")
        self.logs = []
        self.loss_info = collections.deque(maxlen=50)

        #         # Distributed GPU training
        #         if cuda and torch.cuda.device_count() > 1:
        #             print("Using %d GPUS for BERT" % torch.cuda.device_count())
        #             self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_set = train_set
        self.val_set = val_set
        self._set_dataloader()

        self.model = model.to(self.device)

        # Setting the Adam optimizer with hyper-param
        # self.optim = bert_optim.Optimizer(self.model.parameters(),
        # self.model.hidden_size, n_warmup_steps=warmup_steps)
        self.opt = optim.make_optimizer(self.model)

        self.criterion2 = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq
        self.save_freq = save_freq
        self.opt_freq = opt_freq

    def _set_dataloader(self):
        self.train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=16,
            pin_memory=True,
        )

        if self.val_set:
            self.val_loader = DataLoader(
                dataset=self.val_set,
                batch_size=self.batch_size,
                num_workers=16,
                pin_memory=True,
            )

    def train(self, file_path="./check_points/bert/", max_num=200000):
        self.iteration(self.train_loader, file_path, max_num)

    def iteration(self, data_loader, file_path="./check_points/bert/", max_num=200000):
        """
        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """

        data_iter = tqdm.tqdm(enumerate(data_loader))

        total_loss = 0.0
        num = 0
        for i, data in data_iter:
            data = [d.to(self.device) for d in data]
            o = self.model(data[0], data[1])

            loss = self.criterion2(o.transpose(1, 2), data[2])

            loss.backward()
            if i % self.opt_freq == 0:
                self.opt.step()
                self.opt.zero_grad()

            if i % self.log_freq == 0:
                total_loss += loss.item()
                self.loss_info.append(loss.item())
                num += 1
                log = (
                    f"iter: {i}",
                    f"avg_loss : {round(sum(self.loss_info)/len(self.loss_info),3)}",
                    f"loss : {round(loss.item(),3)}",
                )
                self.logs.append(log)
                print(log)

            if i % self.save_freq == 0:
                self.save(i, file_path)

            if i > max_num:
                print("stop training !")
                break

    def save(self, step, file_path="./check_points/bert/"):
        """
        Saving the current BERT model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        self.model.cpu()

        save_file(self.logs, file_path + "logs.pkl")

        output_path = file_path + f"model_{step}.pkl"
        torch.save(self.model.state_dict(), output_path)
        his_path = file_path + f"model_{step-4*self.save_freq}.pkl"
        if os.path.exists(his_path):
            os.remove(his_path)

        output_path = file_path + f"bert_{step}.pkl"
        torch.save(self.model.bert.state_dict(), output_path)
        his_path = file_path + f"bert_{step-4*self.save_freq}.pkl"
        if os.path.exists(his_path):
            os.remove(his_path)

        output_path = file_path + "bert.pkl"
        torch.save(self.model.bert, output_path)

        self.model.to(self.device)
        return output_path
