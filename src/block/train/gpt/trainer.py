import collections
import os
import pickle

import numpy as np

# import optim2
import torch
import tqdm
from torch import nn
from torch.cuda.amp import GradScaler, autocast


def save_file(f, path):
    if os.path.exists(path):
        os.remove(path)
    file = open(path, "wb")
    pickle.dump(f, file)
    file.close()


def _rule(epoch, warmup_steps=500, down_steps=1e6):
    if down_steps < 10 * warmup_steps:
        down_steps = 10 * warmup_steps
    if epoch < warmup_steps:
        lamda = 8 * epoch / warmup_steps
    elif epoch < 2 * warmup_steps:
        lamda = 8 - 7 * (epoch - warmup_steps) / warmup_steps
    elif epoch < down_steps:
        lamda = 1.3 - (epoch - 2 * warmup_steps) / down_steps
    else:
        lamda = 0.3
    return lamda


class Trainer:
    def __init__(
        self,
        train_loader,
        model,
        lr=1e-4,
        batch_size=32,
        cuda: bool = True,
        log_freq: int = 1000,
        save_freq: int = 10000,
        opt_freq=32,
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
        cuda = torch.cuda.is_available() and cuda
        self.device = torch.device("cuda" if cuda else "cpu")
        self.logs = []
        self.loss_info = collections.deque(maxlen=100)

        # Setting the train and test data loader
        self.train_loader = train_loader

        self.model = model.to(self.device)
        self.loss_fct = nn.CrossEntropyLoss(label_smoothing=0.003)
        # Setting the Adam optimizer with hyper-param
        # self.opt = optim2.Optimizer(self.model.parameters())
        # self.opt = optim.make_optimizer(self.model)
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.002
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=_rule)

        self.log_freq = log_freq
        self.save_freq = save_freq
        self.opt_freq = opt_freq
        self.total_loss = []
        self.iter_num = 0

    def log_save(self, iter_num, loss_num, file_path):

        if (iter_num + 1) % self.log_freq == 0:
            self.total_loss.append(loss_num)
            log = (
                f"iter: {self.iter_num}",
                f"avg_loss : {round(np.mean(self.total_loss[-50:]),3)}",
                f"loss : {round(loss_num,3)}",
            )
            self.logs.append(log)
            print(log)

        if (iter_num + 1) % self.save_freq == 0:
            self.save((iter_num + 1), file_path)

    def train(self, file_path="/data/home/ze.song/models/gpt", max_num=1e6):
        self.iteration(
            file_path,
            max_num=max_num,
        )

    def iteration(self, file_path=".", max_num=1e6):
        """
        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """

        scaler = GradScaler()

        data_iter = tqdm.tqdm(self.train_loader)
        iter_num = 0
        for data in data_iter:
            if iter_num > max_num:
                print("training finished!")
                break
            data = [d.to(self.device) for d in data]
            if len(data) == 2:
                inputs, labels = data
                seg = None
            else:
                inputs, seg, labels = data

            with autocast():
                lm_logits = self.model(
                    inputs=inputs,
                    segment_ids=seg,
                )

                # Flatten the tokens
                loss = self.loss_fct(
                    lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
                )
                loss = loss / self.opt_freq
                loss_num = loss.item()

            scaler.scale(loss).backward()

            if (iter_num + 1) % self.opt_freq == 0:
                scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                scaler.step(self.opt)
                scaler.update()
                self.opt.zero_grad()
                self.scheduler.step()

            self.log_save(iter_num, loss_num, file_path)
            iter_num += 1

    def save(self, step, file_path="/data/home/ze.song/models/gpt"):
        """
        Saving the current BERT model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        self.model.cpu()

        save_file(self.logs, file_path + "/logs.pkl")

        output_path = file_path + f"/model_{step}.pkl"
        torch.save(self.model.state_dict(), output_path)
        his_path = file_path + f"/model_{step-5*self.save_freq}.pkl"
        if os.path.exists(his_path):
            os.remove(his_path)

        output_path = file_path + f"/gpt_{step}.pkl"
        torch.save(self.model.model.state_dict(), output_path)
        his_path = file_path + f"/gpt_{step-5*self.save_freq}.pkl"
        if os.path.exists(his_path):
            os.remove(his_path)

        output_path = file_path + f"/lm_{step}.pkl"
        torch.save(self.model.lm_head.state_dict(), output_path)
        his_path = file_path + f"/lm_{step-5*self.save_freq}.pkl"
        if os.path.exists(his_path):
            os.remove(his_path)

        self.model.to(self.device)
        return output_path
