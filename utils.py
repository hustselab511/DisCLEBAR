import torch
import torch.nn.functional as F
import argparse


def L2_Regular(net, ratio):
    loss = 0.
    for param in net.boneage_regression.parameters():
        loss += torch.sum(torch.pow(param, 2)) 
    return ratio * loss


def KL_loss(p, q):
    p_soft = F.softmax(torch.flatten(p, 1), dim=1) + 1e-3
    q_soft = F.softmax(torch.flatten(q, 1), dim=1) + 1e-3

    return torch.sum(p_soft * (p_soft.log() - q_soft.log()))


def attn_kl_loss(t1, t2, t3, t4, s1, s2, s3, s4):
    assert s1.shape == t2.shape
    assert s2.shape == t3.shape

    return KL_loss(t2, s1) + KL_loss(t3, s2)


def get_teacher_args():
    parser = argparse.ArgumentParser(description="Training Teacher Model")
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--in_ch', type=int, help='Number of input channels')
    parser.add_argument('--output_ch', type=int, help='Number of output channels')
    parser.add_argument('--epoch', type=int, help='Number of training epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers')
    parser.add_argument('--img_size', type=int, help='Size of input images')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')

    return parser.parse_args()


def get_phase_I_args():
    parser = argparse.ArgumentParser(description="Training Phase I arguments")
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--img_size', type=int, help='Size of input images')
    parser.add_argument('--data_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--teacher_path', type=str, help='Path to teacher model')
    parser.add_argument('--save_path', type=str, help='Path to save model checkpoints')
    parser.add_argument('--model_name', type=str, help='Model name for saving')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--lr_decay_step', type=int, help='Step for learning rate decay')
    parser.add_argument('--lr_decay_ratio', type=float, help='Learning rate decay ratio')
    parser.add_argument('--weight_decay', type=float, help='Weight decay (L2 regularization)')
    parser.add_argument('--best_loss', type=float, help='Best loss so far')
    parser.add_argument('--attn_loss_ratio', type=int, help='Attention loss ratio')

    return parser.parse_args()


def get_phase_II_args():
    parser = argparse.ArgumentParser(description="Training Phase II arguments")
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--img_size', type=int, help='Size of input images')
    parser.add_argument('--data_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--student_path', type=str, help='Path to the student model checkpoint')
    parser.add_argument('--save_path', type=str, help='Path to save model checkpoints')
    parser.add_argument('--model_name', type=str, help='Model name for saving')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--lr_decay_step', type=int, help='Step for learning rate decay')
    parser.add_argument('--lr_decay_ratio', type=float, help='Learning rate decay ratio')
    parser.add_argument('--weight_decay', type=float, help='Weight decay (L2 regularization)')
    parser.add_argument('--best_loss', type=float, help='Best loss so far')
    parser.add_argument('--wscl_loss_0_ratio', type=float, help='WSCL loss0 ratio')
    parser.add_argument('--wscl_loss_1_ratio', type=float, help='WSCL loss1 ratio')
    parser.add_argument('--WCL_setting', type=str, help='WCL settings in JSON format')


    return parser.parse_args()
