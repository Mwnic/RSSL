import os
import sys
# from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import itertools
import csv

import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
# from tsne import tsne1, tsne3d

from utils.metrics import compute_AUCs, compute_metrics, compute_metrics_test
from sklearn.metrics import confusion_matrix

from utils.metric_logger import MetricLogger
from tsne import tsne1
import pandas as pd

NUM_CLASSES = 7
CLASS_NAMES = ['Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis',
               'Dermatofibroma', 'Vascular lesion']


def epochVal(model, dataLoader, loss_fn, args):
    training = model.training
    model.eval()

    meters = MetricLogger()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()

    gt_study = {}
    pred_study = {}
    studies = []

    with torch.no_grad():
        for i, (study, image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            _, output = model(image)
            # _, output = model(image)

            loss = loss_fn(output, label.clone())
            meters.update(loss=loss)

            output = F.softmax(output, dim=1)

            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])

            # gt = torch.cat((gt, label), 0)
            # pred = torch.cat((pred, output), 0)

        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)

        AUROCs = compute_AUCs(gt, pred, competition=True)

    model.train(training)

    return meters.loss.global_avg, AUROCs


def epochVal_metrics_test(model, dataLoader, epoch):
    training = model.training
    model.eval()

    meters = MetricLogger()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    activation = torch.FloatTensor().cuda()

    gt_study = {}
    pred_study = {}
    activate_study = {}
    weight_study = {}
    y_true = []
    y_score = []
    studies = []

    with torch.no_grad():
        for i, (study, _, image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            activate, output = model(image)
            # weight = wnet(output.softmax(1))
            output = F.softmax(output, dim=1)
            # _, output = model(image)

            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])

                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])
                activate_study[study[i]] = activate[i]
                # weight_study[study[i]] = weight[i]
                y_true.append(label[i].cpu().detach().numpy())
                y_score.append(output[i].cpu().detach().numpy())


        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)
            activation = torch.cat((activation, activate_study[study].view(1, -1)), 0)

        AUROCs, Accus, Senss, Specs, F1 = compute_metrics_test(gt, pred, competition=True)

    model.train(training)

    return AUROCs, Accus, Senss, Specs, F1