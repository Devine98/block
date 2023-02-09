import collections
import os

import conf
import dataset
import funcs
import nets
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.utils import BATCH_COUNT, NUM_SAMPLES


def div_file_list(file_list: list, n: "int>0" = 8, k: "int>0" = 1) -> tuple:
    """
    按file_list中parquet文件的的num_rows切分file_list为n份,并返回第k份和最少rows
    尽可能保证每一份的数量相差不要过大
    """
    length = len(file_list)
    assert length >= n, n >= k
    assert length > 1
    file_list = sorted(file_list)

    def get_file_rows(file):
        return pq.read_metadata(file).num_rows

    nums = [get_file_rows(f) for f in file_list]
    sum_nums = sum(nums)  # 总rows
    if sum_nums <= 0:
        raise ValueError("sum of nums <=0 ")
    avg_nums_per_n = sum_nums / n  # 每份平均rows
    reduce_nums = [nums[0]]  # 每个元素代表这个file和之前所有file的rows的sum
    for i in range(1, length):
        reduce_nums.append(reduce_nums[-1] + nums[i])

    # 根据reduce_nums决定file list中每个file归属哪个k,得到k_list

    def _allocate_k(left_num, right_num, last_k=1):
        k_nums = last_k * avg_nums_per_n
        if right_num <= k_nums:
            return last_k
        r_remaider = right_num - k_nums
        l_remaider = k_nums - left_num
        if r_remaider >= l_remaider:
            return last_k + 1
        else:
            return last_k

    k_list = [1]
    for i in range(1, length):
        k_list.append(_allocate_k(reduce_nums[i - 1], reduce_nums[i], k_list[-1]))
    k_list = np.array(k_list)
    file_list = np.array(file_list)
    nums = np.array(nums)
    n_num = [sum(nums[k_list == i]) for i in range(1, n + 1)]
    min_num = min(n_num)
    res = list(file_list[k_list == k])
    return res, min_num


def get_loader(config: dict, file_list: list, min_num: int):
    """
    从file_list读取不超过min_num的数据构建loader
    """
    cols = config["train_cols"]
    parquet_set = pq.ParquetDataset(file_list)
    dp = parquet_set.read(cols).to_pandas()
    dp = dp[:min_num]
    my_data_set = dataset.DataSet(dp)
    length = len(my_data_set)
    my_loader = torch.utils.data.DataLoader(
        dataset=my_data_set,
        batch_size=config.get("batch_size", 256),
        shuffle=True,
        num_workers=config.get("num_cpus_per_worker", 8),
        pin_memory=True,
    )
    return my_loader, length


def div_loader(path, config, k):
    """
    提取path中的所有parquet文件,并按config.num_workers进行分块读取,返回loader
    """
    file_list = funcs.list_dir(path)
    file_list = [f for f in file_list if f.endswith("parquet")]
    file_slice, min_num = div_file_list(file_list, n=config.get("num_workers", 8), k=k)
    loader, length = get_loader(config, file_slice, min_num)
    return loader, length


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, avg=True):
        self.count += n
        if avg:
            self.val = val
            self.sum += val * n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val


class AverageMeterCollection:
    """
    avg==True,原始形式,用平均后的socre来update
    avg==Flase,求和形式,累加如tp,tn,fp,fn之类的评分,此时,每个worker内部的sum是累加的,但是整个trainer中的sum是每个worker的sum的平均
    """

    def __init__(self):
        self._batch_count = 0
        self.n = 0
        self._meters = collections.defaultdict(AverageMeter)

    def update(self, metrics, n=1, avg=True):
        self._batch_count += 1
        self.n += n
        for metric, value in metrics.items():
            self._meters[metric].update(value, n=n, avg=avg)

    def summary(self, avg=True):
        stats = {BATCH_COUNT: self._batch_count, NUM_SAMPLES: self.n}
        if avg:
            for metric, meter in self._meters.items():
                stats[str(metric)] = meter.avg
                stats["last_" + str(metric)] = meter.val
        else:
            for metric, meter in self._meters.items():
                stats[str(metric)] = meter.sum
        return stats


class TrainingOperator(TrainingOperator):
    def setup(self, config):
        train_loader, val_loader = None, None
        if config["train"] is True:
            train_loader, train_len = div_loader(
                config["train_local_path"], config, k=self._world_rank + 1
            )
            print(self._world_rank, " has train loader :", train_len, " rows")
        if config["val"] is True:
            val_loader, val_len = div_loader(
                config["val_local_path"], config, k=self._world_rank + 1
            )
            print(self._world_rank, " has val loader :", val_len, " rows")
        self.register_data(train_loader=train_loader, validation_loader=val_loader)
        model = nets.make_model(dt=config["dt"])
        optimizer = torch.optim.SGD(
            model.parameters(), config.get("lr", 1e-3), momentum=0.9
        )
        #         criterion = torch.nn.BCEWithLogitsLoss()
        criterion = nets.FocalLoss(gamma=0.75)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        self.model, self.optimizer, self.criterion, self.scheduler = self.register(
            models=model,
            optimizers=optimizer,
            criterion=criterion,
            schedulers=scheduler,
        )
        self.threshold = config.get("threshold", 0.7)

    def train_epoch(self, iterator, info):
        metric_meters = AverageMeterCollection()
        model = self.model
        scheduler = self.scheduler
        criterion = self.criterion
        optimizer = self.optimizer
        model.to(self.device)
        model.train()
        for batch_idx, batch in enumerate(iterator):
            batch = [d.to(self.device) for d in batch]
            y_pred = model(*batch[:-1])
            loss = criterion(y_pred, batch[-1].float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()
            metrics = {"loss": loss.item(), NUM_SAMPLES: batch[0].shape[0]}
            metric_meters.update(metrics, n=metrics.pop(NUM_SAMPLES, 1))
            self.global_step += 1
        scheduler.step()
        return metric_meters.summary()

    def validate(self, val_iterator, info=None):
        model = self.model
        model.eval()
        model.to(self.device)
        confmat = funcs.ConfusionMatrix(num_classes=2)
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_iterator):
                with torch.no_grad():
                    batch = [d.to(self.device) for d in batch]
                    y_pred = (torch.sigmoid(model(*batch[:-1])) > self.threshold).long()
                    y = batch[-1].long()
                    confmat.update(y, y_pred)
            confmat.reduce_from_all_processes()
        return confmat


@funcs.clock
def get_trainer(config):
    def initialization_hook():
        os.environ["NCCL_DEBUG"] = "WARNING"

    trainer = TorchTrainer(
        training_operator_cls=TrainingOperator,
        scheduler_step_freq="epoch",
        config=config,
        initialization_hook=initialization_hook,
        add_dist_sampler=False,
        num_workers=config.get("num_workers", 8),
        num_cpus_per_worker=config.get("num_cpus_per_worker", 12),
        use_gpu=config.get("use_gpu", True),
        use_tqdm=False,
        backend="gloo",
    )
    return trainer


@funcs.clock
def train_model(trainer, epochs=12, val=True, val_epoch=3, model_path=conf.model_path):
    best_score = 0
    model = None
    for i in range(epochs):
        stats = trainer.train()
        print(stats)
        if val and i % val_epoch == (val_epoch - 1):
            confmat = trainer.validate(reduce_results=False)[0]
            confmat.compute()
            f1 = confmat.f1
            if f1 >= best_score:
                print("f1 : ", round(f1, 4), " >= best_score : ", round(best_score, 4))
                best_score = f1
                model = trainer.get_model()
                torch.save(model.state_dict(), model_path)
    #         trainer.save("/data/home/ze.song/trainer.checkpoint")
    if model is None:
        model = trainer.get_model()
    return model, best_score
