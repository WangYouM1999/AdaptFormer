import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from .soft_ce import SoftCrossEntropyLoss
from .joint_loss import JointLoss
from .dice import DiceLoss


class EdgeLoss(nn.Module):
    def __init__(self, ignore_index=255, edge_factor=10.0):
        super(EdgeLoss, self).__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.edge_factor = edge_factor

    def get_boundary(self, x):
        laplacian_kernel_target = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).cuda(device=x.device)
        x = x.unsqueeze(1).float()
        x = F.conv2d(x, laplacian_kernel_target, padding=1)
        x = x.clamp(min=0, max=1.0)
        x[x >= 0.1] = 1
        x[x < 0.1] = 0

        return x

    def compute_edge_loss(self, logits, targets):
        bs = logits.size()[0]
        boundary_targets = self.get_boundary(targets)
        boundary_targets = boundary_targets.view(bs, 1, -1)
        # print(boundary_targets.shape)
        logits = F.softmax(logits, dim=1).argmax(dim=1).squeeze(dim=1)
        boundary_pre = self.get_boundary(logits)
        boundary_pre = boundary_pre / (boundary_pre + 0.01)
        # print(boundary_pre)
        boundary_pre = boundary_pre.view(bs, 1, -1)
        # print(boundary_pre)
        edge_loss = F.binary_cross_entropy_with_logits(boundary_pre, boundary_targets)

        return edge_loss

    def forward(self, logits, targets):
        loss = self.main_loss(logits, targets) + self.compute_edge_loss(logits, targets) * self.edge_factor
        return loss


class OHEM_CELoss(nn.Module):

    def __init__(self, thresh=0.7, ignore_index=255):
        super(OHEM_CELoss, self).__init__()
        self.thresh = thresh
        self.ignore_index = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        thresh = -torch.log(torch.tensor(self.thresh, requires_grad=False, dtype=torch.float)).cuda(device=logits.device)
        n_min = labels[labels != self.ignore_index].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, reduction='mean', thresh=0.7, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)     # 概率小于阈值的挖出来
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

class AdaptFormerLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
        # edge_loss
        self.edge_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

    def forward(self, logits, labels):

        if self.training and len(logits) == 3:
            logit_main, logit_aux, logit_distance = logits
            # loss_list = []

            loss1 = self.main_loss(logit_main, labels)
            # loss_list.append(loss1.item())

            loss2 = self.edge_loss(logit_aux, labels)
            # loss_list.append(loss2.item())

            h, w = logit_distance.size()[-2:]
            edge_labels = labels.unsqueeze(1).to(torch.float32)
            edge_labels = F.interpolate(edge_labels, size=(h, w), mode='bilinear', align_corners=False).to(torch.int64)
            edge_labels = edge_labels.squeeze(1)
            loss3 = self.edge_loss(logit_distance, edge_labels)
            # loss_list.append(loss3.item())
            loss = loss1 + loss2 + loss3
            # loss_list.append(loss.item())
            return loss
        else:
            loss = self.main_loss(logits, labels)
            return loss


class FTAdaptFormerLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        # edge_loss
        self.edge_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

    def forward(self, logits, labels):

        if self.training and len(logits) == 2:
            logit_main, logit_edge = logits

            loss1 = self.main_loss(logit_main, labels)

            h, w = logit_edge.size()[-2:]
            edge_labels = labels.unsqueeze(1).to(torch.float32)
            edge_labels = F.interpolate(edge_labels, size=(h, w), mode='bilinear', align_corners=False).to(torch.int64)
            edge_labels = edge_labels.squeeze(1)
            loss2 = self.edge_loss(logit_edge, edge_labels)
            loss = loss1 + loss2

            return loss
        else:
            loss = self.main_loss(logits, labels)
            return loss


class BuildUFormerLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

        self.edge_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

    def forward(self, logits, labels):
        if self.training and len(logits) == 3:
            logit_main, logit_aux, logit_distance = logits
            # loss_list = []

            loss1 = self.main_loss(logit_main, labels)
            # loss_list.append(loss1.item())

            loss2 = self.edge_loss(logit_aux, labels)
            # loss_list.append(loss2.item())

            h, w = logit_distance.size()[-2:]
            edge_labels = labels.unsqueeze(1).to(torch.float32)
            edge_labels = F.interpolate(edge_labels, size=(h, w), mode='bilinear', align_corners=False).to(torch.int64)
            edge_labels = edge_labels.squeeze(1)
            loss3 = self.edge_loss(logit_distance, edge_labels)
            # loss_list.append(loss3.item())
            loss = loss1 + loss2 + loss3
            return loss
        else:
            loss = self.main_loss(logits, labels)
            return loss


class UNetFormerLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
    def forward(self, logits, labels):
        if self.training and len(logits) == 2:
            logit_main, logit_aux = logits
            loss = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(logit_aux,labels)
        else:
            loss = self.main_loss(logits, labels)
        return loss


class FTUNetFormerLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
        # edge_loss
        self.edge_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

    def forward(self, logits, labels):

        if self.training and len(logits) == 3:
            logit_main, logit_aux, logit_distance = logits
            # logit_aux = logit_aux.squeeze(1)
            loss_list = []

            loss1 = self.main_loss(logit_main, labels)
            loss_list.append(loss1.item())

            loss2 = self.edge_loss(logit_aux, labels)
            loss_list.append(loss2.item())

            h, w = logit_distance.size()[-2:]
            edge_labels = labels.unsqueeze(1).to(torch.float32)
            edge_labels = F.interpolate(edge_labels, size=(h, w), mode='bilinear', align_corners=False).to(torch.int64)
            edge_labels = edge_labels.squeeze(1)
            loss3 = self.edge_loss(logit_distance, edge_labels)
            loss_list.append(loss3.item())

            loss = loss1 + loss2 + loss3
            loss_list.append(loss.item())
            return loss, loss_list
        else:
            loss = self.main_loss(logits, labels)
            return loss

if __name__ == '__main__':
    targets = torch.randint(low=0, high=2, size=(2, 16, 16))
    logits = torch.randn((2, 2, 16, 16))
    # print(targets)
    model = EdgeLoss()
    loss = model.compute_edge_loss(logits, targets)

    print(loss)