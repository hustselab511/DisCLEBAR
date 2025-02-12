import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from student_model_p1 import get_student_p1


class SODA(nn.Module):
    def __init__(self, in_channels, attn_dim, in_size) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.attn_dim = attn_dim
        self.scale = attn_dim ** -0.5
        self.in_size = in_size

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.q = nn.Linear(in_channels+32, attn_dim, bias=False)
        self.k = nn.Linear(in_channels+32, attn_dim, bias=False)
        self.v = nn.Linear(in_channels+32, attn_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(attn_dim, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(attn_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, gender_encode):
        B, C, H, W = x.shape
        cls_token = self.avg_pool(x)
        cls_token = rearrange(cls_token, 'b d h w -> b (h w) d')

        feature_vector = rearrange(x, 'b d h w -> b (h w) d')

        feature_total = torch.cat((cls_token, feature_vector), dim=1)
        gender_encode = gender_encode.unsqueeze(dim=1).repeat(1, (H*W)+1, 1)
        feature_total = torch.cat((feature_total, gender_encode), dim=-1)

        q = self.norm(self.relu(self.q(feature_total)))
        k = self.norm(self.relu(self.k(feature_total)))
        v = self.norm(self.relu(self.v(feature_total)))

        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = self.softmax(attn * self.scale)

        feature_out = torch.matmul(attn, v)

        cls_token = feature_out[:, 0].reshape(B, -1, 1, 1)

        cls_token = self.fc2(cls_token)

        attn = rearrange(attn[:, 0, 1:], 'b (h w) -> b h w', h=self.in_size, w=self.in_size).unsqueeze(dim=1)

        return attn * x, torch.flatten(cls_token, 1), attn


class Student_Model_P2(nn.Module):
    def __init__(self, model_p1):
        super(Student_Model_P2, self).__init__()

        self.ResBlock0 = model_p1.ResBlock0
        self.attn0 = model_p1.attn0
        self.ResBlock1 = model_p1.ResBlock1
        self.attn1 = model_p1.attn1
        self.freeze_params()

        self.ResBlock2 = model_p1.ResBlock2
        self.attn2 = SODA(1024, 768, 32)
        self.ResBlock3 = model_p1.ResBlock3
        self.attn3 = SODA(2048, 768, 16)

        self.gender_encoder = model_p1.gender_encoder

        self.boneage_regression = model_p1.boneage_regression

        self.cls_Embedding_0 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )

        self.cls_Embedding_1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )

    def forward(self, image, gender):
        gender_encode = self.gender_encoder(gender)
        x0, attn0 = self.attn0(self.ResBlock0(image))
        x1, attn1 = self.attn1(self.ResBlock1(x0))
        x2, cls_token2, attn2 = self.attn2(self.ResBlock2(x1), gender_encode)
        x3, cls_token3, attn3 = self.attn3(self.ResBlock3(x2), gender_encode)

        x = F.adaptive_avg_pool2d(x3, 1)
        x = torch.flatten(x, 1)

        x = torch.cat([x, gender_encode], dim=1)

        cls_token2 = F.normalize(self.cls_Embedding_0(cls_token2), dim=1)
        cls_token3 = F.normalize(self.cls_Embedding_1(cls_token3), dim=1)

        x = self.boneage_regression(x)

        return x, cls_token2, cls_token3, attn0, attn1, attn2, attn3

    def freeze_params(self):
        for _, param in self.ResBlock0.named_parameters():
            param.requires_grad = False
        for _, param in self.attn0.named_parameters():
            param.requires_grad = False
        for _, param in self.ResBlock1.named_parameters():
            param.requires_grad = False
        for _, param in self.attn1.named_parameters():
            param.requires_grad = False


def get_student_p2(student_path):
    model_p1 = get_student_p1()
    if student_path is not None:
        model_p1.load_state_dict(torch.load(student_path))
    return Student_Model_P2(model_p1)

