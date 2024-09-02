import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    loss = 0
    weight = 1.

    dx, dy = gradient(pred_map)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
    return loss


class PoissonGradientLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """L_{grad} = \frac{1}{2hw}\sum_{m=1}^{H}\sum_{n=1}{W}(\partial f(I_{Blend}) - 
                        (\partial f(I_{Source}) + \partial f(I_{Target})))_{mn}^2

           See **Deep Image Blending** for detail.
        """
        super(PoissonGradientLoss, self).__init__()
        self.reduction = reduction

    def forward(self, source, target, blend, mask):
        f = torch.Tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1, 1, 3, 3).to(target)
        f = f.repeat((3, 1, 1, 1))
        grad_s = F.conv2d(source, f, padding=1, groups=3) * mask
        grad_t = F.conv2d(target, f, padding=1, groups=3) * (1 - mask)
        grad_b = F.conv2d(blend, f, padding=1, groups=3)
        return nn.MSELoss(reduction=self.reduction)(grad_b, (grad_t + grad_s))


class GradientLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                            f'Supported ones are: {_reduction_modes}')

    def forward(self, pred, target):
        _, cin, _, _ = pred.shape
        _, cout, _, _ = target.shape
        assert cin == 3 and cout == 3
        kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
                        [1, 0, -1]]).view(1, 1, 3, 3).to(target)
        ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
                        [-1, -2, -1]]).view(1, 1, 3, 3).to(target)
        kx = kx.repeat((3, 1, 1, 1))
        ky = ky.repeat((3, 1, 1, 1))

        pred_grad_x = F.conv2d(pred, kx, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, ky, padding=1, groups=3)
        target_grad_x = F.conv2d(target, kx, padding=1, groups=3)
        target_grad_y = F.conv2d(target, ky, padding=1, groups=3)

        loss = (
            nn.L1Loss(reduction=self.reduction)(
                pred_grad_x, target_grad_x) +
            nn.L1Loss(reduction=self.reduction)(
                pred_grad_y, target_grad_y))
        return loss * self.loss_weight


class L_TV(nn.Module):
    def __init__(self):
        super(L_TV, self).__init__()
    def forward(self, x):
        _, _, h, w = x.size()
        count_h = (h - 1) * w
        count_w = (w - 1) * h

        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w - 1], 2).sum()
        return (h_tv / count_h + w_tv / count_w) / 2.0



class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=[0.25,0.75], gamma=2, ignore_index=None, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

        if self.alpha is None:
            self.alpha = torch.ones(2)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = np.asarray(self.alpha)
            self.alpha = np.reshape(self.alpha, (2))
            assert self.alpha.shape[0] == 2, \
                'the `alpha` shape is not match the number of class'
        elif isinstance(self.alpha, (float, int)):
            self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        else:
            raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target):
        prob = torch.clamp(output, self.smooth, 1.0 - self.smooth)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_loss = -self.alpha[0] * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob) * pos_mask
        neg_loss = -self.alpha[1] * torch.pow(prob, self.gamma) * \
                   torch.log(torch.sub(1.0, prob)) * neg_mask

        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
        num_neg = neg_mask.view(neg_mask.size(0), -1).sum()

        if num_pos == 0:
            loss = neg_loss
        else:
            loss = pos_loss / num_pos + neg_loss / num_neg
        return loss

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

def style_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        _, c, w, h = A_feat.size()
        A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
        B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
        A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
        B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
        loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
    return loss_value

def TV_loss(x):
    h_x = x.size(2)
    w_x = x.size(3)
    h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
    w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
    return h_tv + w_tv

def perceptual_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        loss_value += torch.mean(torch.abs(A_feat - B_feat))
    return loss_value