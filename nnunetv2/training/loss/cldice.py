import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import torch.nn.functional as F
from typing import Callable

class SoftClDiceLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, cl_kwargs, weight_cl=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param cl_kwargs:
        :param weight_cl:
        :param weight_dice:
        """
        super(SoftClDiceLoss, self).__init__()
        if ignore_label is not None:
            dc_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_cl = weight_cl
        self.ignore_label = ignore_label

        self.cl = soft_cldice(apply_nonlin=softmax_helper_dim1, **cl_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_ClDC_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        cl_loss = self.cl(net_output, target_dice) \
            if self.weight_cl != 0 and (self.ignore_label is None or num_fg > 0) else 0 #TODO: je sploh reasonable ignorirat label?

        #result = self.weight_cl * cl_loss + self.weight_dice * dc_loss
        #return torch.stack((self.weight_cl*cl_loss,  1+self.weight_dice*dc_loss)) #cat([self.weight_cl*cl_loss.unsqueeze(0), self.weight_dice*dc_loss.unsqueeze(0)])
        return torch.stack((self.weight_cl*(cl_loss-1),  self.weight_dice*dc_loss)) #cat([self.weight_cl*cl_loss.unsqueeze(0), self.weight_dice*dc_loss.unsqueeze(0)])


        

class soft_cldice(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, iter_: int = 3, smooth: float = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin

    def forward(self, net_output, gt):
       # print("out and gt shapes:", net_output.shape, gt.shape)
        with torch.no_grad():
            if net_output.ndim != gt.ndim:
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if net_output.shape == gt.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                y_onehot = torch.zeros(net_output.shape, device=net_output.device)
                y_onehot.scatter_(1, gt.long(), 1)
        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        skel_pred = soft_skel(net_output, self.iter)
        skel_true = soft_skel(y_onehot, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_onehot)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, net_output)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice
    


def soft_erode(img):
    if len(img.shape)==4:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    elif len(img.shape)==5:
        p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
        p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
        p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape)==4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape)==5:
        return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1  =  soft_open(img)
    skel  =  F.relu(img-img1)
    for j in range(iter_):
        img  =  soft_erode(img)
        img1  =  soft_open(img)
        delta  =  F.relu(img-img1)
        skel  =  skel +  F.relu(delta-skel*delta)
    return skel


