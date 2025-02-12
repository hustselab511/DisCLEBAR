import os
import random
import time
import warnings
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset

from datasets import RSNATrainDataset, RSNAValidDataset
from utils import L1_Regular, get_phase_II_args

from student_model_p2 import get_student_p2
from wscl import WSCL

warnings.filterwarnings("ignore")

args = get_phase_II_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_fn(train_loader, loss_fn, wscl_fn, optimizer):
    contrast_model.train()
    training_loss = 0
    total_wscl_loss_0 = 0
    total_wscl_loss_1 = 0
    total_size = 0
    for idx, data in enumerate(train_loader):
        image, gender = data[0]
        image = image.type(torch.FloatTensor).cuda()
        gender = gender.type(torch.FloatTensor).cuda()
        img_gt = data[2].type(torch.FloatTensor).cuda()

        batch_size = len(data[1])
        label = data[1].type(torch.FloatTensor).cuda()

        optimizer.zero_grad()

        class_feature, cls_token0, cls_token1, _, _, _, _ = contrast_model(image, gender)
        y_pred = class_feature.squeeze()
        label = label.squeeze()

        loss = loss_fn(y_pred, label)
        wscl_loss_0 = wscl_fn(cls_token0, img_gt, gender)
        wscl_loss_1 = wscl_fn(cls_token1, img_gt, gender)

        penalty_loss = L1_Regular(contrast_model, 1e-5)
        total_loss = loss + penalty_loss
        total_loss = total_loss + wscl_loss_0.squeeze(0) * args.wscl_loss_0_ratio
        total_loss = total_loss + wscl_loss_1.squeeze(0) * args.wscl_loss_1_ratio
        total_loss.backward()

        optimizer.step()
        batch_loss = loss.item()
        print(f"batch_loss: {batch_loss}, WCL0: {wscl_loss_0.item()}, WCL1: {wscl_loss_1.item()}, "
              f"penalty_loss: {penalty_loss.item()}")

        training_loss += batch_loss
        total_size += batch_size
        total_wscl_loss_0 += wscl_loss_0.item()
        total_wscl_loss_1 += wscl_loss_1.item()

    return training_loss, total_wscl_loss_0, total_wscl_loss_1, total_size


def evaluate_fn(val_loader):
    contrast_model.eval()

    mae_loss = 0
    val_total_size = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            val_total_size += len(data[1])

            image, gender = data[0]
            image = image.type(torch.FloatTensor).cuda()
            gender = gender.type(torch.FloatTensor).cuda()

            label = data[1].cuda()

            class_feature, cls_token0, cls_token1, s1, s2, s3, s4 = contrast_model(image, gender)
            y_pred = (class_feature * boneage_div) + boneage_mean
            y_pred = y_pred.squeeze()
            label = label.squeeze()
            batch_loss = F.l1_loss(y_pred, label, reduction='sum').item()

            mae_loss += batch_loss
    return mae_loss, val_total_size


def training_start(args):
    best_loss = float('inf')
    loss_fn = nn.L1Loss(reduction='sum')
    wcl_setting = json.loads(args.WCL_setting)
    wscl_fn = WSCL(p=wcl_setting['p'], tempS=wcl_setting['tempS'], thresholdS=wcl_setting['thresholdS'], tempW=wcl_setting['tempW'])

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, contrast_model.parameters()),
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_ratio)

    for epoch in range(args.num_epochs):
        print(f"epoch {epoch + 1}")

        start_time = time.time()
        training_loss, training_wscl_loss_0, training_wscl_loss_1, total_size = train_fn(train_loader, loss_fn, wscl_fn, optimizer)

        valid_mae_loss, val_total_size = evaluate_fn(valid_loader)

        training_mean_loss = training_loss / total_size
        mean_wscl_loss_0, mean_wscl_loss_1 = training_wscl_loss_0 / total_size, training_wscl_loss_1 / total_size
        valid_mean_mae = valid_mae_loss / val_total_size
        if valid_mean_mae < best_loss:
            best_loss = valid_mean_mae
            torch.save(contrast_model.state_dict(), '/'.join([save_path, f'{model_name}.bin']))
            args.best_loss = best_loss
            with open(os.path.join(save_path, 'setting.txt'), 'w') as f:
                f.writelines('------------------ start ------------------' + '\n')
                for key, value in vars(args).items():
                    f.write(f"{key}: {value}\n")
                f.writelines('------------------- end -------------------')
        scheduler.step()
        print(f"Training Loss: {training_mean_loss}, Mean WSCL 0: {mean_wscl_loss_0}, Mean WSCL 1: "
              f"{mean_wscl_loss_1}, Validation Loss: {valid_mean_mae}, Cost Time: {round(time.time() - start_time, 2)}, LR: {optimizer.param_groups[0]['lr']}")

    print(f'best loss: {best_loss}')


if __name__ == "__main__":
    model_name = args.model_name
    save_path = os.path.join(args.save_path, model_name)
    os.makedirs(save_path, exist_ok=True)
    contrast_model = get_student_p2(student_path=args.student_path).cuda()
    data_dir = args.data_dir

    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")

    train_csv = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_csv)
    valid_csv = os.path.join(data_dir, "valid.csv")
    valid_df = pd.read_csv(valid_csv)

    boneage_mean = train_df['boneage'].mean()
    boneage_div = train_df['boneage'].std()
    print(f"boneage_mean is {boneage_mean}")
    print(f"boneage_div is {boneage_div}")
    print(f'{save_path} start')

    train_set = RSNATrainDataset(train_df, train_path, boneage_mean, boneage_div, args.img_size)
    valid_set = RSNAValidDataset(valid_df, valid_path, boneage_mean, boneage_div, args.img_size)

    print(train_set.__len__())

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    training_start(args)
