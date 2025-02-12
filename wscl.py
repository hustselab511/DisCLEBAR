import torch
import torch.nn as nn
import torch.nn.functional as F


class WSCL(nn.Module):
    def __init__(self, p, tempS, thresholdS, tempW):
        super(WSCL, self).__init__()

        self.p = p
        self.tempS = tempS
        self.tempS_Male = tempS
        self.tempS_Female = tempS / 2
        self.thresholdS = thresholdS

        self.tempW = tempW

        self.criterion = nn.MSELoss(reduction='sum')

    def count_score_in(self, label, gender):
        gender = gender.view(-1)
        tempS_gender = gender * self.tempS_Male + (1 - gender) * self.tempS_Female

        length = len(label)
        label_clone_1 = label.clone().view(length, 1)
        label_clone_2 = label.clone().view(1, length)
        tempS_gender = tempS_gender.view(length, 1)

        distance_matrix = torch.abs(label_clone_1 - label_clone_2)

        score_matrix = torch.exp(-(torch.div(distance_matrix, tempS_gender)).pow(self.p))
        score_matrix = score_matrix * (score_matrix >= self.thresholdS)

        one_hot_gender = F.one_hot(gender.type(torch.LongTensor), num_classes=2).squeeze().float().cuda()
        gender_mask = torch.matmul(one_hot_gender, one_hot_gender.t())
        score_matrix = score_matrix * gender_mask
        return score_matrix

    def count_distance_in(self, logit):
        logit_clone = logit.clone()
        dot = torch.exp(torch.matmul(logit, logit_clone.T) / self.tempW)
        dot_sum = dot.sum(-1, keepdim=True)
        dot_matrix = torch.clamp(torch.div(dot, dot_sum), min=1e-10)
        return dot_matrix

    def forward(self, minibatch_features, label, gender):
        score_matrix = self.count_score_in(label, gender)
        dot_matrix = self.count_distance_in(minibatch_features)
        weight_dot_matrix = (score_matrix * dot_matrix).sum(-1)
        weight_dot_matrix = - torch.log(torch.clamp(weight_dot_matrix, min=1e-10))
        loss_triplet = weight_dot_matrix.sum()
        return loss_triplet
