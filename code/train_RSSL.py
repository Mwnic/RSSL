import os
import sys
import shutil
import argparse
import logging
import time
import random
import math
import numpy as np
from tensorboardX import SummaryWriter

import torch
import csv
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn import preprocessing

from networks.models import DenseNet121
from utils import losses, ramps
from utils.metrics import compute_AUCs
from utils.metric_logger import MetricLogger
from dataloaders import dataset
from dataloaders.dataset import TwoStreamBatchSampler
from utils.util import get_timestamp
from validation import epochVal, epochVal_metrics_test
from wideresnet import WNet

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/skin/training_data/', help='dataset root dir')
parser.add_argument('--csv_file_pre_train', type=str, default='../data/skin/training.csv', help='training set csv file')
parser.add_argument('--csv_file_train', type=str, default='../data/skin/training.csv', help='training set csv file')
parser.add_argument('--csv_file_val', type=str, default='../data/skin/validation.csv', help='validation set csv file')
parser.add_argument('--csv_file_test', type=str, default='../data/skin/testing.csv', help='testing set csv file')
parser.add_argument('--exp', type=str, default='xxxx', help='model_name')
parser.add_argument('--epochs', type=int, default=180, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=4, help='number of labeled data per batch')
parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model')
parser.add_argument('--labeled_num', type=int, default=1400, help='number of labeled')
parser.add_argument('--base_lr', type=float, default=1e-4, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
### tune
parser.add_argument('--resume', type=str, default=None,
                    help='model to resume')
# parser.add_argument('--resume', type=str,  default=None, help='GPU to use')
parser.add_argument('--start_epoch', type=int, default=0, help='start_epoch')
parser.add_argument('--global_step', type=int, default=0, help='global_step')
### costs
parser.add_argument('--label_uncertainty', type=str, default='U-Ones', help='label type')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=30, help='consistency_rampup')
parser.add_argument('--pre_consistency_rampup', type=float, default=40, help='consistency_rampup')
# D3L
parser.add_argument('--meta_lr', type=float, default=0.001)
parser.add_argument('--lr_wnet', type=float,
                    default=3e-5)  # this parameter need to be carefully tuned for different settings

parser.add_argument('--Temperature', default=0.5, type=float, help='temperature for sharpening')
parser.add_argument('--pre_lr', type=float, default=1e-3, help='learning rate on pre_training')
parser.add_argument('--pre_epochs', type=int, default=80, help='')
parser.add_argument('--pre_ema_decay', type=float, default=0.999, help='ema_decay')

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * 4
base_lr = args.base_lr
labeled_bs = args.labeled_bs * 4

if args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

if torch.cuda.is_available():
    device = "cuda"


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242

    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_current_consistency_weight_pre(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.pre_consistency_rampup)


def update_model_variables(pre_model, model):
    # Use the true average until the exponential average is more correct
    alpha = 0.
    for param, pre_param in zip(model.parameters(), pre_model.parameters()):
        param.data.mul_(alpha).add_(1 - alpha, pre_param.data)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if __name__ == "__main__":
    ## make logging file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + './checkpoint')
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(ema=False):
        # Network definition
        net = DenseNet121(out_size=dataset.N_CLASSES, mode=args.label_uncertainty, drop_rate=args.drop_rate)
        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    model = create_model()
    ema_model = create_model(ema=True)
    pre_model = create_model()
    pre_ema_model = create_model(ema=True)
    wnet = WNet(7, 100, 1).to(device)
    pre_optimizer = torch.optim.SGD(pre_model.parameters(), lr=args.pre_lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
    optimizer_wnet = torch.optim.Adam(wnet.params(), lr=args.lr_wnet)

    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        args.global_step = checkpoint['global_step']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # dataset
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    pre_train_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                                csv_file=args.csv_file_pre_train,
                                                transform=dataset.TransformTwice(transforms.Compose([
                                                    transforms.Resize((224, 224)),
                                                    transforms.RandomCrop(size=224,
                                                                          padding=int(224 * 0.0625),
                                                                          padding_mode='reflect'),
                                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                                                    transforms.ToTensor(),
                                                    normalize, ])))

    train_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                            csv_file=args.csv_file_train,
                                            transform=dataset.TransformTwice(transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                                transforms.RandomHorizontalFlip(),
                                                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                # transforms.RandomRotation(10),
                                                # transforms.RandomResizedCrop(224),
                                                transforms.ToTensor(),
                                                normalize,
                                            ])))

    val_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                          csv_file=args.csv_file_val,
                                          transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))
    test_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                           csv_file=args.csv_file_test,
                                           transform=transforms.Compose([
                                               transforms.Resize((224, 224)),
                                               transforms.ToTensor(),
                                               normalize,
                                           ]))

    labeled_idxs = list(range(args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, 7000))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    pre_train_dataloader = DataLoader(dataset=pre_train_dataset, batch_size=128,  # sampler=pre_train_sampler,
                                      num_workers=0, pin_memory=True, shuffle=True, worker_init_fn=worker_init_fn)
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler,
                                  num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=0, pin_memory=True)  # , worker_init_fn=worker_init_fn)

    model.train()
    pre_model.train()
    wnet.train()

    loss_fn = losses.cross_entropy_loss()
    PGC_loss = losses.PseudoGroupContrast()
    PGC_loss_w = losses.PseudoGroupContrast_w()
    PGC_loss_pre = losses.PseudoGroupContrast_pre()

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = args.global_step
    lr_ = base_lr
    pre_lr_ = args.pre_lr
    model.train()

    # train
    for epoch in range(args.start_epoch, args.epochs):
        meters_loss = MetricLogger(delimiter="  ")
        meters_loss_classification = MetricLogger(delimiter="  ")
        meters_loss_consistency = MetricLogger(delimiter="  ")
        meters_loss_contrastive = MetricLogger(delimiter="  ")
        meters_loss_contrastive_l = MetricLogger(delimiter="  ")
        meters_loss_pl = MetricLogger(delimiter="  ")
        time1 = time.time()
        if epoch <= args.pre_epochs:
            iter_max = len(pre_train_dataloader)
        else:
            iter_max = len(train_dataloader)

        weight_study = []

        if epoch <= args.pre_epochs:
            for i, (_, _, (image_batch, ema_image_batch), _) in enumerate(pre_train_dataloader):
                time2 = time.time()
                # print('fetch data cost {}'.format(time2-time1))
                image_batch, ema_image_batch = image_batch.cuda(), ema_image_batch.cuda()

                inputs = image_batch
                ema_inputs = ema_image_batch

                activations, outputs = pre_model(inputs)

                with torch.no_grad():
                    ema_activations, ema_output = pre_ema_model(ema_inputs)
                ## calculate the loss

                pseudo_label = torch.softmax(ema_output, dim=-1)
                _, un_labels = torch.max(pseudo_label, dim=-1)

                loss = PGC_loss_pre(activations, ema_activations, un_labels)

                pre_optimizer.zero_grad()
                loss.backward()
                pre_optimizer.step()
                update_ema_variables(pre_model, pre_ema_model, args.pre_ema_decay, iter_num)
                update_model_variables(pre_model, model)
                update_model_variables(pre_ema_model, ema_model)

                meters_loss.update(loss=loss)

                iter_num = iter_num + 1

                # write tensorboard
                if i % 50 == 0:
                    writer.add_scalar('lr', lr_, iter_num)
                    writer.add_scalar('loss/loss', loss, iter_num)

                    logging.info(
                        "\nEpoch: {}, iteration: {}/{}, ==> train <===, loss: {:.6f}, lr: {}"
                            .format(epoch, i, iter_max, meters_loss.loss.avg,
                                    pre_optimizer.param_groups[0]['lr']))

                    image = inputs[-1, :, :]
                    grid_image = make_grid(image, 5, normalize=True)
                    writer.add_image('raw/Image', grid_image, iter_num)

                    image = ema_inputs[-1, :, :]
                    grid_image = make_grid(image, 5, normalize=True)
                    writer.add_image('noise/Image', grid_image, iter_num)

        else:
            for i, (study, _, (image_batch, ema_image_batch), label_batch) in enumerate(train_dataloader):
                time2 = time.time()

                meta_net = create_model()
                meta_net.load_state_dict(model.state_dict())

                image_batch, ema_image_batch, label_batch = image_batch.cuda(), ema_image_batch.cuda(), label_batch.cuda()

                ema_inputs = ema_image_batch
                inputs = image_batch
                _, lb_labels = torch.max(label_batch, dim=1)

                activations, outputs = meta_net(inputs)
                with torch.no_grad():
                    ema_activations, ema_output = ema_model(ema_inputs)
                    ema_output_detach = ema_output.detach()

                weight = wnet(outputs.softmax(1)[labeled_bs:])
                norm_weight = torch.sum(weight)

                ## calculate the loss
                loss_classification = loss_fn(outputs[:labeled_bs], label_batch[:labeled_bs])
                loss = loss_classification

                ## MT loss (have no effect in the beginneing)
                if args.ema_consistency == 1:
                    consistency_weight = get_current_consistency_weight(epoch - args.pre_epochs - 1)
                    consistency_dist_l = torch.sum(losses.softmax_mse_loss(outputs[:labeled_bs], ema_output[
                                                                                                 :labeled_bs])) / labeled_bs  # / dataset.N_CLASSES
                    consistency_dist_ul = torch.sum(weight *losses.softmax_mse_loss(outputs[labeled_bs:], ema_output[#
                                                                                                           labeled_bs:])) / norm_weight  # / dataset.N_CLASSES
                    consistency_dist = consistency_dist_ul + consistency_dist_l
                    consistency_loss = consistency_weight * consistency_dist

                    contrastive_weight = consistency_weight

                    pseudo_label = torch.softmax(outputs[labeled_bs:], dim=-1)
                    _ , un_labels = torch.max(pseudo_label, dim=-1)
                    p_loss = F.cross_entropy(pseudo_label, lb_labels[labeled_bs:].long(), reduction='none')
                    p_loss =0.1 * ((weight * torch.unsqueeze(p_loss, dim=1)).mean()) * consistency_weight
                    labels = torch.cat((lb_labels[:labeled_bs], un_labels))

                    contrastive_dist_ul = 0.01 * PGC_loss_w(activations[labeled_bs:], ema_activations[labeled_bs:],
                                                             labels[labeled_bs:], weight)
                    lab_weight = torch.unsqueeze(torch.ones(labeled_bs).float(), dim=1).cuda()
                    contrastive_dist_l = 0.05 * PGC_loss_w(activations[:labeled_bs], ema_activations[:labeled_bs],
                                                           labels[:labeled_bs], lab_weight)

                    contrastive_dist = contrastive_dist_ul + contrastive_dist_l
                    contrastive_loss = contrastive_weight * contrastive_dist


                else:
                    consistency_loss = 0.0
                    contrastive_loss = 0.0
                    consistency_weight = 0.0
                    consistency_dist = 0.0
                # + consistency_loss

                if (args.ema_consistency == 1):  #
                    loss = loss_classification + contrastive_loss + consistency_loss + p_loss

                meta_net.zero_grad()
                grads = torch.autograd.grad(loss, (meta_net.params()), create_graph=True, allow_unused=True)#.module
                meta_net.update_params(lr_inner=args.meta_lr, source_params=grads)#.module
                # update_ema_variables(meta_net,ema_meta_net, args.ema_decay, iter_num)
                del grads

                activations_hat, y_g_hat = meta_net(inputs)
                # ema_activations_hat, y_ema_hat = ema_meta_net(ema_inputs)

                l_s_meta = loss_fn(y_g_hat[:labeled_bs], label_batch[:labeled_bs])
                l_g_meta = l_s_meta

                optimizer_wnet.zero_grad()
                l_g_meta.backward()
                optimizer_wnet.step()

                activations, outputs = model(inputs)

                with torch.no_grad():
                    ema_activations, ema_output = ema_model(ema_inputs)
                    weight = wnet(outputs.softmax(1)[labeled_bs:])
                    for z in range(len(weight)):
                        weight_study.append(
                            [study[labeled_bs + z], torch.argmax(label_batch[labeled_bs + z], dim=0), weight[z]])
                        # weight_study[study[labeled_bs + z]] = weight[z]
                    norm_weight = torch.sum(weight)

                loss_classification = loss_fn(outputs[:labeled_bs], label_batch[:labeled_bs])
                loss = loss_classification

                if args.ema_consistency == 1:
                    # consistency_weight = get_current_consistency_weight(epoch - args.pre_epochs - 1)
                    consistency_dist_l = torch.sum(losses.softmax_mse_loss(outputs[:labeled_bs], ema_output[
                                                                                                 :labeled_bs])) / labeled_bs  # / dataset.N_CLASSES
                    consistency_dist_ul = torch.sum(weight *losses.softmax_mse_loss(outputs[labeled_bs:], ema_output[#
                                                                                                           labeled_bs:])) / norm_weight  # / dataset.N_CLASSES
                    consistency_dist = consistency_dist_ul + consistency_dist_l
                    consistency_loss = consistency_weight * consistency_dist


                    pseudo_label = torch.softmax(outputs[labeled_bs:], dim=-1)
                    _, un_labels = torch.max(pseudo_label, dim=-1)
                    p_loss = F.cross_entropy(pseudo_label, lb_labels[labeled_bs:].long(), reduction='none')
                    p_loss = 0.1*(( weight *torch.unsqueeze(p_loss, dim=1)).mean())*consistency_weight#
                    labels = torch.cat((lb_labels, un_labels))

                    contrastive_dist_ul = 0.01 * PGC_loss(activations[labeled_bs:], ema_activations[labeled_bs:],
                                                           labels[labeled_bs:], weight)
                    lab_weight = torch.unsqueeze(torch.ones(labeled_bs).float(), dim=1).cuda()
                    contrastive_dist_l = 0.05 * PGC_loss(activations[:labeled_bs], ema_activations[:labeled_bs],
                                                         labels[:labeled_bs], lab_weight)
                    contrastive_dist = contrastive_dist_ul + contrastive_dist_l
                    contrastive_loss = contrastive_weight * contrastive_dist
                else:
                    consistency_loss = 0.0
                    contrastive_loss = 0.0
                    consistency_weight = 0.0
                    consistency_dist = 0.0

                if (epoch > args.pre_epochs + 20) and (args.ema_consistency == 1):  #
                    loss = loss_classification + contrastive_loss + consistency_loss + p_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                update_ema_variables(model, ema_model, args.ema_decay, iter_num)

                # outputs_soft = F.softmax(outputs, dim=1)
                meters_loss.update(loss=loss)
                meters_loss_classification.update(loss=loss_classification)
                meters_loss_consistency.update(loss=consistency_loss)
                meters_loss_contrastive.update(loss=contrastive_dist_ul)
                meters_loss_contrastive_l.update(loss=contrastive_dist_l)
                meters_loss_pl.update(loss =p_loss)

                iter_num = iter_num + 1

                # write tensorboard
                if i % 80 == 0:
                    writer.add_scalar('lr', lr_, iter_num)
                    writer.add_scalar('loss/loss', loss, iter_num)
                    writer.add_scalar('loss/loss_classification', loss_classification, iter_num)
                    writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
                    writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
                    writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

                    logging.info(
                        "\nEpoch: {}, iteration: {}/{}, ==> train <===, loss: {:.6f}, classification loss: {:.6f}, consistency loss: {:.6f},contrastive_l_loss: {:.6f}, contrastive_loss: {:.6f}, pl_loss: {:.6f},consistency weight: {:.6f}, lr: {}"
                            .format(epoch, i, iter_max, meters_loss.loss.avg, meters_loss_classification.loss.avg,
                                    meters_loss_consistency.loss.avg, meters_loss_contrastive_l.loss.avg,
                                    meters_loss_contrastive.loss.avg,meters_loss_pl.loss.avg, consistency_weight,
                                    optimizer.param_groups[0]['lr']))

                    image = inputs[-1, :, :]
                    grid_image = make_grid(image, 5, normalize=True)
                    writer.add_image('raw/Image', grid_image, iter_num)

                    image = ema_inputs[-1, :, :]
                    grid_image = make_grid(image, 5, normalize=True)
                    writer.add_image('noise/Image', grid_image, iter_num)

        timestamp = get_timestamp()

        AUROCs, Accus, Senss, Specs, F1 = epochVal_metrics_test(model, val_dataloader, epoch)
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()
        F1_avg = np.array(F1).mean()

        logging.info("\nVAL Student: Epoch: {}, iteration: {}".format(epoch, i))
        logging.info("\nVAL AUROC: {:6f}, VAL Accus: {:6f}, VAL Senss: {:6f}, VAL Specs: {:6f}, VAL F1: {:6f}"
                     .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg))
        logging.info(
            "AUROCs: " + " ".join(["{}:{:.6f}".format(dataset.CLASS_NAMES[i], v) for i, v in enumerate(AUROCs)]))

        # test student
        #
        AUROCs, Accus, Senss, Specs, F1 = epochVal_metrics_test(model, test_dataloader, epoch)
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()
        F1_avg = np.array(F1).mean()

        logging.info("\nTEST Student: Epoch: {}, iteration: {}".format(epoch, i))
        logging.info("\nTEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, TEST F1: {:6f}"
                     .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg))
        logging.info(
            "AUROCs: " + " ".join(["{}:{:.6f}".format(dataset.CLASS_NAMES[i], v) for i, v in enumerate(AUROCs)]))

        # save model
        save_mode_path = os.path.join(snapshot_path + 'checkpoint/', 'epoch_' + str(epoch + 1) + '.pth')
        torch.save({'epoch': epoch + 1,
                    'global_step': iter_num,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epochs': epoch,
                    # 'AUROC'     : AUROC_best,
                    }
                   , save_mode_path
                   )
        logging.info("save model to {}".format(save_mode_path))

        # update learning rate
        if epoch <= args.pre_epochs:
            lr_min = 0
            pre_lr_ = math.fabs(lr_min + (1 + math.cos(1 * (epoch - 20) * math.pi / 100)) * (pre_lr_ - lr_min) / 2.)
            for param_group in pre_optimizer.param_groups:
                param_group['lr'] = pre_lr_
        else:
            # lr_ = lr_ * 0.9
            lr_min = 5e-9
            lr_ = math.fabs(
                lr_min + (1 + math.cos(1 * ((epoch - args.pre_epochs - 1) - 20) * math.pi / 100)) * (lr_ - lr_min) / 2.)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num + 1) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
