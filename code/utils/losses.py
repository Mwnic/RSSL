import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from collections import Counter

"""
The different uncertainty methods loss implementation.
Including:
    Ignore, Zeros, Ones, SelfTrained, MultiClass
"""

METHODS = ['U-Ignore', 'U-Zeros', 'U-Ones', 'U-SelfTrained', 'U-MultiClass']
CLASS_NUM = [1113, 6705, 514, 327, 1099, 115, 142]
CLASS_WEIGHT = torch.Tensor([10000 / (i) for i in CLASS_NUM]).cuda()


class Loss_Zeros(object):
    """
    map all uncertainty values to 0
    """

    def __init__(self):
        self.base_loss = torch.nn.BCELoss(reduction='mean')

    def __call__(self, output, target):
        target[target == -1] = 0
        return self.base_loss(output, target)


class Loss_Ones(object):
    """
    map all uncertainty values to 1
    """

    def __init__(self):
        self.base_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def __call__(self, output, target):
        target[target == -1] = 1
        return self.base_loss(output, target)


class cross_entropy_loss(object):
    """
    map all uncertainty values to a unique value "2"
    """

    def __init__(self):
        self.base_loss = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT, reduction='mean')

    def __call__(self, output, target):
        # target[target == -1] = 2
        output_softmax = F.softmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        return self.base_loss(output_softmax, target.long())


def get_UncertaintyLoss(method):
    assert method in METHODS

    if method == 'U-Zeros':
        return Loss_Zeros()

    if method == 'U-Ones':
        return Loss_Ones()


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2 * CLASS_WEIGHT
    return mse_loss


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

class PseudoGroupContrast(nn.Module):
    def __init__(self):
        super(PseudoGroupContrast, self).__init__()
        # 特征大小
        self.projector_dim = 128
        # 分类数量
        self.class_num = 7
        # 列表长度
        self.queue_size = 168
        # 申请长期缓存，随机初始值正则化
        self.register_buffer("queue_list", torch.randn(self.queue_size * self.class_num, self.projector_dim))
        self.register_buffer("queue_weight", torch.zeros(self.queue_size * self.class_num, 1))
        self.queue_list = F.normalize(self.queue_list, dim=1).cuda()
        self.queue_weight = self.queue_weight.cuda()  # F.normalize(self.queue_weight, dim=0).cuda()
        # 温度参数
        self.temperature = 0.5
        # 初始化标记
        self.init_flag = [True for i in range(self.class_num)]

    @torch.no_grad()
    def _dequeue_and_enqueue(self, ema_feature, label, weight):
        # 当前类列表
        temp_list = self.queue_list[label * self.queue_size:(label + 1) * self.queue_size, :]
        temp_weight = self.queue_weight[label * self.queue_size:(label + 1) * self.queue_size, :]
        # 把当前特征拼接到队列中，再从头取相同大小的队列
        temp_list = torch.cat([ema_feature, temp_list], dim=0)
        temp_list = temp_list[0:self.queue_size, :]
        temp_weight = torch.cat([weight, temp_weight], dim=0)
        temp_weight = temp_weight[0:self.queue_size, :]
        # 替换原队列
        self.queue_list[label * self.queue_size:(label + 1) * self.queue_size, :] = temp_list
        self.queue_weight[label * self.queue_size:(label + 1) * self.queue_size, :] = temp_weight

    def forward(self, activation, ema_activation, pseudo_label, weight):
        # 正则化
        feature = F.normalize(activation, dim=1)
        # feature = weight * feature
        ema_feature = F.normalize(ema_activation, dim=1)
        # ema_feature = weight * ema_feature
        # one-hot伪标转换为数字型
        label = pseudo_label  # torch.argmax(pseudo_label, dim=1)
        # 输入图像数量
        batch_size = feature.size(0)
        # 对比损失
        contrast_loss = 0.0

        # feature和ema_feature对应位置计算点乘
        # 使用einsum代替dot提高计算效率
        l_pos = torch.einsum('nl,nl->n', [feature, ema_feature])
        weight_pos = torch.squeeze(weight, dim=1)
        #weight_pos = torch.einsum('nl,nl->n', [weight, weight])
        # l_pos = torch.exp(l_pos/temperature)

        # 获取当前列表
        current_queue_list = self.queue_list.clone().detach()
        current_queue_weight = self.queue_weight.clone().detach()

        logits_list = torch.Tensor([]).cuda()

        # 计算所有样本
        for i in range(batch_size):
            # 当前样本
            current_f = feature[i:i + 1]
            current_weight = weight[i:i + 1]
            current_ema_f = ema_feature[i:i + 1]
            current_c = label[i]
            ith_ema = l_pos[i:i + 1]
            ith_weight = weight_pos[i:i + 1]

            # 第一次出现某类样本

            # if self.init_flag[current_c] is True:
            #     # 用当前特征代替随机初始化
            #     for j in range(self.queue_size):
            #         self._dequeue_and_enqueue(current_ema_f, current_c)
            #     # 标记为假
            #     self.init_flag[current_c] = False

            # 构造正样本和负样本列表
            pos_sample = current_queue_list[current_c * self.queue_size:(current_c + 1) * self.queue_size, :]
            pos_sample_weight = current_queue_weight[current_c * self.queue_size:(current_c + 1) * self.queue_size, :]
            neg_sample = torch.cat([current_queue_list[0:current_c * self.queue_size, :],
                                    current_queue_list[(current_c + 1) * self.queue_size:, :]], dim=0)
            neg_sample_weight = torch.cat([current_queue_weight[0:current_c * self.queue_size, :],
                                           current_queue_weight[(current_c + 1) * self.queue_size:, :]], dim=0)

            # 计算正样本
            ith_pos = torch.einsum('nl,nl->n', [current_f, pos_sample])
            pos_weight = torch.einsum('nl,nl->n',[current_weight,pos_sample_weight])
            #pos_weight = torch.squeeze(current_weight * pos_sample_weight, dim=1)
            ith_pos = torch.exp(ith_pos / self.temperature)  # current_weight * pos_sample_weight *
            # 正样本列表D+1
            # ith_pos = torch.cat([ith_ema, ith_pos], dim=0)
            pos = torch.sum(ith_pos)
            # 计算负样本
            ith_neg = torch.einsum('nl,nl->n', [current_f, neg_sample])
            neg_weight = torch.squeeze(current_weight * neg_sample_weight, dim=1)
            ith_neg = torch.exp(ith_neg / self.temperature)
            neg = torch.sum(ith_neg)

            # 计算当前样本的对比损失
            # contrast = ith_pos/(ith_ema + pos + neg)
            ith_pos_all = torch.cat([ith_ema, ith_pos], dim=0)
            ith_weight_all = torch.cat([ith_weight, pos_weight], dim=0)
            contrast = ith_pos_all / (ith_ema + pos + neg)
            contrast = ith_weight_all * (-torch.log(contrast + 1e-8))
            contrast = torch.sum(contrast) / (self.queue_size + 1)
            # 损失累加
            contrast_loss = contrast_loss + contrast

            # 更新队列
            self._dequeue_and_enqueue(current_ema_f, current_c, current_weight)

        # 当前batch的pgc-loss均值
        pgc_loss = contrast_loss / batch_size

        # pgc_logits = nn.LogSoftmax(dim=1)(logits_list/self.temperature)
        #
        # pgc_labels = torch.zeros(batch_size, 1+self.queue_size*self.class_num).cuda()
        # pgc_labels[:, 0:1+self.queue_size].fill_(1.0/(1+self.queue_size))
        #
        # loss_fn = nn.KLDivLoss(reduction='batchmean')
        # pgc_loss = loss_fn(pgc_logits, pgc_labels)

        # 返回pgc-loss
        return pgc_loss


class PseudoGroupContrast_w(nn.Module):
    def __init__(self):
        super(PseudoGroupContrast_w, self).__init__()
        # 特征大小
        self.projector_dim = 128
        # 分类数量
        self.class_num = 7
        # 列表长度
        self.queue_size = 168
        # 申请长期缓存，随机初始值正则化
        self.register_buffer("queue_list", torch.randn(self.queue_size * self.class_num, self.projector_dim))
        self.register_buffer("queue_weight", torch.zeros(self.queue_size * self.class_num, 1))
        self.queue_list = F.normalize(self.queue_list, dim=1).cuda()
        self.queue_weight = self.queue_weight.cuda()# F.normalize(self.queue_weight, dim=0).cuda()
        # 温度参数
        self.temperature = 0.5
        # 初始化标记
        self.init_flag = [True for i in range(self.class_num)]

    @torch.no_grad()
    def _dequeue_and_enqueue(self, ema_feature, label, weight):
        # 当前类列表
        temp_list = self.queue_list[label * self.queue_size:(label + 1) * self.queue_size, :]
        temp_weight = self.queue_weight[label * self.queue_size:(label + 1) * self.queue_size, :]
        # 把当前特征拼接到队列中，再从头取相同大小的队列
        temp_list = torch.cat([ema_feature, temp_list], dim=0)
        temp_list = temp_list[0:self.queue_size, :]
        temp_weight = torch.cat([weight, temp_weight], dim=0)
        temp_weight = temp_weight[0:self.queue_size, :]
        # 替换原队列
        self.queue_list[label * self.queue_size:(label + 1) * self.queue_size, :] = temp_list
        self.queue_weight[label * self.queue_size:(label + 1) * self.queue_size, :] = temp_weight

    def forward(self, activation, ema_activation, pseudo_label, weight):
        # 正则化
        feature =  F.normalize(activation, dim=1)
        #feature = weight * feature
        ema_feature = F.normalize(ema_activation, dim=1)
        #ema_feature = weight * ema_feature
        # one-hot伪标转换为数字型
        label = pseudo_label  # torch.argmax(pseudo_label, dim=1)
        # 输入图像数量
        batch_size = feature.size(0)
        # 对比损失
        contrast_loss = 0.0

        # feature和ema_feature对应位置计算点乘
        # 使用einsum代替dot提高计算效率
        l_pos = torch.einsum('nl,nl->n', [feature, ema_feature])
        weight_pos = torch.squeeze(weight, dim=1) #torch.einsum('nl,nl->n', [weight, weight])
        # l_pos = torch.exp(l_pos/temperature)

        # 获取当前列表
        current_queue_list = self.queue_list.clone().detach()
        current_queue_weight = self.queue_weight.clone().detach()

        logits_list = torch.Tensor([]).cuda()

        # 计算所有样本
        for i in range(batch_size):
            # 当前样本
            current_f = feature[i:i + 1]
            current_weight = weight[i:i+1]
            current_ema_f = ema_feature[i:i + 1]
            current_c = label[i]
            ith_ema = l_pos[i:i + 1]
            ith_weight = weight_pos[i:i + 1]

            # 第一次出现某类样本

            # if self.init_flag[current_c] is True:
            #     # 用当前特征代替随机初始化
            #     for j in range(self.queue_size):
            #         self._dequeue_and_enqueue(current_ema_f, current_c)
            #     # 标记为假
            #     self.init_flag[current_c] = False

            # 构造正样本和负样本列表
            pos_sample = current_queue_list[current_c * self.queue_size:(current_c + 1) * self.queue_size, :]
            pos_sample_weight = current_queue_weight[current_c * self.queue_size:(current_c + 1) * self.queue_size, :]
            neg_sample = torch.cat([current_queue_list[0:current_c * self.queue_size, :],
                                    current_queue_list[(current_c + 1) * self.queue_size:, :]], dim=0)
            neg_sample_weight = torch.cat([current_queue_weight[0:current_c * self.queue_size, :],
                                     current_queue_weight[(current_c + 1) * self.queue_size:, :]], dim=0)

            # 计算正样本
            ith_pos = torch.einsum('nl,nl->n', [current_f, pos_sample])
            pos_weight = torch.einsum('nl,nl->n',[current_weight,pos_sample_weight])
            ith_pos = torch.exp(ith_pos / self.temperature)# current_weight * pos_sample_weight *
            # 正样本列表D+1
            # ith_pos = torch.cat([ith_ema, ith_pos], dim=0)
            pos = torch.sum(ith_pos)
            # 计算负样本
            ith_neg = torch.einsum('nl,nl->n', [current_f, neg_sample])
            neg_weight = torch.squeeze(current_weight * neg_sample_weight, dim=1)
            ith_neg = torch.exp(ith_neg / self.temperature)
            neg = torch.sum(ith_neg)

            # 计算当前样本的对比损失
            # contrast = ith_pos/(ith_ema + pos + neg)
            ith_pos_all = torch.cat([ith_ema, ith_pos], dim=0)
            ith_weight_all = torch.cat([ith_weight, pos_weight], dim=0)
            contrast = ith_pos_all / (ith_ema + pos + neg)
            contrast = ith_weight_all*(-torch.log(contrast + 1e-8))
            contrast = torch.sum(contrast) / (self.queue_size + 1)
            # 损失累加
            contrast_loss = contrast_loss + contrast

            # 更新队列
            self._dequeue_and_enqueue(current_ema_f, current_c, current_weight)

            # 当前batch的pgc-loss均值
        pgc_loss = contrast_loss / batch_size

        # pgc_logits = nn.LogSoftmax(dim=1)(logits_list/self.temperature)
        #
        # pgc_labels = torch.zeros(batch_size, 1+self.queue_size*self.class_num).cuda()
        # pgc_labels[:, 0:1+self.queue_size].fill_(1.0/(1+self.queue_size))
        #
        # loss_fn = nn.KLDivLoss(reduction='batchmean')
        # pgc_loss = loss_fn(pgc_logits, pgc_labels)

        # 返回pgc-loss
        return pgc_loss


class PseudoGroupContrast_pre(nn.Module):
    def __init__(self):
        super(PseudoGroupContrast_pre, self).__init__()
        # 特征大小
        self.projector_dim = 128
        # 分类数量
        self.class_num = 7
        # 列表长度
        self.queue_size = 168
        # 申请长期缓存，随机初始值正则化
        self.register_buffer("queue_list", torch.randn(self.queue_size * self.class_num, self.projector_dim))
        self.queue_list = F.normalize(self.queue_list, dim=1).cuda()
        # 温度参数
        self.temperature = 0.5
        # 初始化标记
        self.init_flag = [True for i in range(self.class_num)]

    @torch.no_grad()
    def _dequeue_and_enqueue(self, ema_feature, label):
        # 当前类列表
        temp_list = self.queue_list[label * self.queue_size:(label + 1) * self.queue_size, :]
        # 把当前特征拼接到队列中，再从头取相同大小的队列
        temp_list = torch.cat([ema_feature, temp_list], dim=0)
        temp_list = temp_list[0:self.queue_size, :]
        # 替换原队列
        self.queue_list[label * self.queue_size:(label + 1) * self.queue_size, :] = temp_list

    def forward(self, activation, ema_activation, pseudo_label):
        # 正则化
        feature = F.normalize(activation, dim=1)
        # feature = weight * feature
        ema_feature = F.normalize(ema_activation, dim=1)
        # ema_feature = weight*ema_feature
        # one-hot伪标转换为数字型
        label = pseudo_label  # torch.argmax(pseudo_label, dim=1)
        # 输入图像数量
        batch_size = feature.size(0)
        # 对比损失
        contrast_loss = 0.0

        # feature和ema_feature对应位置计算点乘
        # 使用einsum代替dot提高计算效率
        l_pos = torch.einsum('nl,nl->n', [feature, ema_feature])
        # l_pos = torch.exp(l_pos/temperature)

        # 获取当前列表
        current_queue_list = self.queue_list.clone().detach()

        logits_list = torch.Tensor([]).cuda()

        # 计算所有样本
        for i in range(batch_size):
            # 当前样本
            current_f = feature[i:i + 1]
            current_ema_f = ema_feature[i:i + 1]
            current_c = label[i]
            ith_ema = l_pos[i:i + 1]

            # 第一次出现某类样本

            # if self.init_flag[current_c] is True:
            #     # 用当前特征代替随机初始化
            #     for j in range(self.queue_size):
            #         self._dequeue_and_enqueue(current_ema_f, current_c)
            #     # 标记为假
            #     self.init_flag[current_c] = False

            # 构造正样本和负样本列表
            pos_sample = current_queue_list[current_c * self.queue_size:(current_c + 1) * self.queue_size, :]
            neg_sample = torch.cat([current_queue_list[0:current_c * self.queue_size, :],
                                    current_queue_list[(current_c + 1) * self.queue_size:, :]], dim=0)

            # 计算正样本
            ith_pos = torch.einsum('nl,nl->n', [current_f, pos_sample])
            ith_pos = torch.exp(ith_pos / self.temperature)
            # 正样本列表D+1
            # ith_pos = torch.cat([ith_ema, ith_pos], dim=0)
            pos = torch.sum(ith_pos)
            # 计算负样本
            ith_neg = torch.einsum('nl,nl->n', [current_f, neg_sample])
            ith_neg = torch.exp(ith_neg / self.temperature)
            neg = torch.sum(ith_neg)

            # 计算当前样本的对比损失
            # contrast = ith_pos/(ith_ema + pos + neg)
            ith_pos_all = torch.cat([ith_ema, ith_pos], dim=0)
            contrast = ith_pos_all / (ith_ema + pos + neg)
            contrast = -torch.log(contrast + 1e-8)
            contrast = torch.sum(contrast) / (self.queue_size + 1)
            # 损失累加
            contrast_loss = contrast_loss + contrast

            # 更新队列
            self._dequeue_and_enqueue(current_ema_f, current_c)

        # 当前batch的pgc-loss均值
        pgc_loss = contrast_loss / batch_size

        # pgc_logits = nn.LogSoftmax(dim=1)(logits_list/self.temperature)
        #
        # pgc_labels = torch.zeros(batch_size, 1+self.queue_size*self.class_num).cuda()
        # pgc_labels[:, 0:1+self.queue_size].fill_(1.0/(1+self.queue_size))
        #
        # loss_fn = nn.KLDivLoss(reduction='batchmean')
        # pgc_loss = loss_fn(pgc_logits, pgc_labels)

        # 返回pgc-loss
        return pgc_loss


