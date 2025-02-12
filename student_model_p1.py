import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


def get_pretrained_resnet50(pretrained=True):
    model = resnet50(pretrained=pretrained)
    fc_in_features = model.fc.in_features
    model = list(model.children())[:-2]
    return model, fc_in_features


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_ch, in_ch // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_ch // ratio, in_ch, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_ch, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_ch, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        attn_ca = self.ca(x)
        out = x * attn_ca
        attn_sa = self.sa(out)
        result = out * attn_sa
        return result, attn_sa


class Student_Model_P1(nn.Module):
    def __init__(self, ge_ch, resnet_list, fc_in_features):
        super(Student_Model_P1, self).__init__()
        self.fc_in_features = fc_in_features
        self.ResBlock0 = nn.Sequential(*resnet_list[0:5])
        self.ResBlock0[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.attn0 = CBAM(in_ch=256, ratio=8, kernel_size=3)
        self.ResBlock1 = resnet_list[5]
        self.attn1 = CBAM(in_ch=512, ratio=8, kernel_size=3)
        self.ResBlock2 = resnet_list[6]
        self.attn2 = CBAM(in_ch=1024, ratio=16, kernel_size=3)
        self.ResBlock3 = resnet_list[7]
        self.attn3 = CBAM(in_ch=2048, ratio=16, kernel_size=3)

        self.gender_encoder = nn.Sequential(
            nn.Linear(1, ge_ch),
            nn.BatchNorm1d(ge_ch),
            nn.ReLU()
        )

        self.boneage_regression = nn.Sequential(
            nn.Linear(fc_in_features + ge_ch, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, image, gender):
        x0, attn0 = self.attn0(self.ResBlock0(image))
        x1, attn1 = self.attn1(self.ResBlock1(x0))
        x2, attn2 = self.attn2(self.ResBlock2(x1))
        x3, attn3 = self.attn3(self.ResBlock3(x2))

        x = F.adaptive_avg_pool2d(x3, 1)
        x = torch.flatten(x, 1)

        gender_encode = self.gender_encoder(gender)
        x = torch.cat([x, gender_encode], dim=1)

        x = self.boneage_regression(x)

        return x, attn0, attn1, attn2, attn3


def get_student_p1(pretrained=True):
    return Student_Model_P1(32, *get_pretrained_resnet50(pretrained=pretrained))
