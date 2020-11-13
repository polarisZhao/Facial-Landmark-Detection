import torch
from torch import nn
import math
import numpy as np
import cv2
from torch.nn import functional as F


class WingLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(WingLoss, self).__init__()
        self.w = 10.0
        self.epsilon = 2.0
        self.reduction = reduction

    def forward(self, pred, landmark):
        diff_abs = torch.abs(pred - landmark)
        c = self.w * (1.0 - math.log(1.0 + self.w / self.epsilon))
        ret = torch.where(self.w > diff_abs,
                          self.w * torch.log(1.0 + diff_abs / self.epsilon),
                          diff_abs - c)
        ret = torch.mean(ret) if self.reduction == 'mean' else ret
        return ret


class AWing(nn.Module):
    def __init__(self,
                 alpha=2.1,
                 omega=14.,
                 epsilon=1.,
                 theta=0.5,
                 reduction='mean'):
        super(AWing, self).__init__()
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta
        self.reduction = reduction

    def forward(self, pred, landmark):
        diff_abs = torch.abs(pred - landmark)

        theta_div_epsilon = self.theta / self.epsilon
        alpha_sub_landmark = self.alpha - landmark

        T1 = theta_div_epsilon**alpha_sub_landmark
        T2 = theta_div_epsilon**(alpha_sub_landmark - 1)
        A = self.omega * 1 / (1 + T1) * alpha_sub_landmark * T2 / self.epsilon
        C = self.theta * A - self.omega * torch.log1p(T1)

        ret = torch.where(
            diff_abs < self.theta,
            self.omega * torch.log1p(
                (diff_abs / self.epsilon)**alpha_sub_landmark),
            A * diff_abs - C)

        ret = torch.mean(ret) if self.reduction == 'mean' else ret
        return ret


class WeightedLoss(nn.Module):
    def __init__(self, W=10., reduction='mean'):
        super().__init__()
        self.W = W
        self.Awing = AWing(reduction=False)
        self.reduction = reduction

    def _generate_weight_map(self, heatmap, k_size=3):
        n, c, h, w = heatmap.size()
        mask = torch.zeros_like(heatmap)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for i in range(n):
            img_list = []
            for j in range(c):
                img_list.append(np.round(heatmap[i][j].cpu().numpy() * 255))
            img_merge = cv2.merge(img_list)
            img_dilate = cv2.morphologyEx(img_merge, cv2.MORPH_DILATE, kernel)
            img_dilate[img_dilate < h * 0.2] = 0
            img_dilate[img_dilate >= w * 0.2] = 1
            img_dilate = np.array(img_dilate, dtype=np.int)
            img_dilate = img_dilate.transpose(2, 0, 1)
            mask[i] = torch.from_numpy(img_dilate)
        return mask

    def forward(self, pred, landmark):
        M = self._generate_weight_map(landmark)
        ret = self.Awing(pred, landmark) * (self.W * M + 1.)
        ret = torch.mean(ret) if self.reduction == 'mean' else ret
        return ret


# if __name__ == "__main__":
#     x1 = torch.rand(2, 3, 256, 256)
#     x2 = torch.rand(2, 3, 256, 256)
#     losses = WeightedLoss()
#     y = losses(x1, x2)
#     print(y)