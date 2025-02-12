import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import unets
import numpy as np
import random

from unet_dataset import SegmentationDataset
from utils import get_teacher_args

args = get_teacher_args()


seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
    if type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)


def train(net, train_dataloader, valid_dataloader, device, num_epoch, lr, init=True):
    if init:
        net.apply(init_xavier)
    print('training on:', device)
    net.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    best_loss = 1000
    for epoch in range(num_epoch):
        print("Epoch: {}".format(epoch + 1))
        net.train()

        train_loss = 0.0
        for data, label in tqdm(train_dataloader):
            data, label = data.to(device), label.to(device)
            predict = net(data)
            predict = torch.nn.functional.sigmoid(predict)
            predict_flat = predict.view(predict.size(0), -1)
            label_flat = label.view(label.size(0), -1)

            loss = criterion(predict_flat, label_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epoch}], Train Loss: {train_loss:.4f}')

        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, label in valid_dataloader:
                data, label = data.to(device), label.to(device)
                predict = net(data)

                predict = torch.nn.functional.sigmoid(predict)
                predict_flat = predict.view(predict.size(0), -1)
                label_flat = label.view(label.size(0), -1)
                loss = criterion(predict_flat, label_flat)
                test_loss += loss.item()

            test_loss = test_loss / len(valid_dataloader)
            print(f'Epoch [{epoch + 1}/{num_epoch}],Test Loss: {test_loss:.4f}')
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(net.state_dict(), os.path.join(model_save_path, model_save_name))
    print(f"best loss: {best_loss:.4f}")


if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((800, 800), scale=(0.5, 1.0)),
        transforms.RandomAffine(degrees=(10, 20), translate=(0.1, 0.2)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    train_image_dir = '../data/train'
    train_label_dir = '../data/train_labels'

    test_image_dir = '../data/val'
    test_label_dir = '../data/val_labels'

    trainDataset = SegmentationDataset(train_image_dir, train_label_dir, transform=transform_train)
    trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testDataset = SegmentationDataset(test_image_dir, test_label_dir, transform=transform_val)
    testLoader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    segment_model = unets.Attn_UNet(img_ch=args.in_ch, output_ch=args.output_ch).cuda()
    model_save_path = './unet_ckp'
    os.makedirs(model_save_path, exist_ok=True)
    model_save_name = "unet.pth"

    train(segment_model, trainLoader, testLoader, device=torch.device('cuda:0'), num_epoch=args.epoch, lr=args.lr)
