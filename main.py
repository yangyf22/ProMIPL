#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from __future__ import print_function
import os
import math
import time
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import scipy.io as io
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score


from model import VAEAttention
from utils import device, logging, seed_everything, args, parser
from dataloader import load_data_mat, load_idx_mat, MIPLDataloader


def evaluate_label(loader, model):
    '''
    model testing
    '''
    model.eval()
    all_true_bag_lab = []
    all_pred_bag_lab = []
    all_pred_bag_prob = np.empty((0, args.nr_class))
    for data, y_can, true_bag_lab, _ in loader:
        data = data.to(device)
        true_bag_lab = true_bag_lab.to(device)
        data = data.to(torch.float32)
        true_bag_lab = true_bag_lab.to(torch.float32)
        output = model.evaluate_objective(data)
        all_pred_bag_prob = np.vstack((all_pred_bag_prob, output.detach().cpu().numpy()))
        _, pred_bag_lab = torch.max(output.data, 1)
        all_true_bag_lab.append(true_bag_lab.item())
        all_pred_bag_lab.append(pred_bag_lab.item())
    all_true_bag_lab = np.array(all_true_bag_lab)
    all_pred_bag_lab = np.array(all_pred_bag_lab)
    acc = accuracy_score(all_true_bag_lab, all_pred_bag_lab)

    return acc


def train_procedure(model, optimizer, loader, epoch):
    '''
     model training
    '''
    model.train()
    train_loss = 0.
    for index, (data, partial_bag_lab, true_bag_lab, batch_idx) in enumerate(loader):
        if args.cuda:
            data, partial_bag_lab, true_bag_lab = data.cuda(), partial_bag_lab.cuda(), true_bag_lab.cuda()
        data, partial_bag_lab, true_bag_lab = Variable(data), Variable(partial_bag_lab), Variable(true_bag_lab)
        data = data.to(torch.float32)
        partial_bag_lab = partial_bag_lab.to(torch.float32)
        true_bag_lab = true_bag_lab.to(torch.float32)
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, new_partial_bag_lab = model.calculate_objective(data, partial_bag_lab)
        train_loss += loss.item()
        alpha_value = alpha_list[epoch]
        new_partial_bag_lab = new_partial_bag_lab.cpu().detach().numpy()
        partial_bag_lab = partial_bag_lab.cpu().detach().numpy()
        new_label = alpha_value * partial_bag_lab + (1. - alpha_value) * new_partial_bag_lab
        new_label = np.squeeze(new_label, axis=0)
        loader.dataset.partial_bag_lab_list[batch_idx] = new_label
        # backward pass
        loss.backward()
        optimizer.step()
    ## calculate loss and error for epoch
    train_loss /= len(loader)
    if (epoch) % 10 == 0:
        logging.info('Epoch: {}, Train loss: {:.4f}.'.format(epoch+1, train_loss))
    return model, loader


def adjust_alpha(epochs):
    '''
    the weighting factor in Equation (8)
    $\alpha^{(t)} = \frac{T-t}{T}$
    '''
    alpha_list = [1.0] * epochs
    for ep in range(epochs):
        alpha_list[ep] = (epochs - ep) / (epochs)
    return alpha_list


def run():
    seed_everything(args.seed) 
    accuracy = np.empty((num_trial, num_fold))
    for trial_i in range(num_trial):
        for fold_i in range(num_fold):
            logging.info('\t---------------- time: {}, fold: {} ----------------'.format(trial_i + 1, fold_i + 1))
            idx_file = index_path + '/' + all_folds[fold_i]
            # load the index and dataset
            idx_tr, idx_te = load_idx_mat(idx_file)
            train_loader = data_utils.DataLoader(
                MIPLDataloader(all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab_processed, idx_tr,
                            args.nr_fea, args.ds, seed=args.seed, train=True, normalize=args.normalize),
                batch_size=args.bs, shuffle=True, **loader_kwargs)
            test_loader = data_utils.DataLoader(
                MIPLDataloader(all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab_processed, idx_te,
                            args.nr_fea, args.ds, seed=args.seed, train=False, normalize=args.normalize),
                batch_size=args.bs, shuffle=False, **loader_kwargs)

            # ---------------- init model ----------------
            model = VAEAttention(args)
            if args.cuda:
                model.cuda()
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.reg)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            

            # -------------- start training ---------------
            for epoch in range(args.epochs):
                model, train_loader = train_procedure(model, optimizer, train_loader, epoch)
                model.decay_temperature()
                if args.lr_sche:
                    lr_scheduler.step()
            # -------------- start testing ---------------
            test_accuracy = evaluate_label(test_loader, model)
            logging.info('test_acc: {:.3f}'.format(test_accuracy))
            accuracy[trial_i, fold_i] = test_accuracy
    logging.info('The mean and std of accuracy at {} times {} folds: {}, {}'.format(
        num_trial, num_fold, np.around(np.mean(accuracy), 3), np.around(np.std(accuracy), 3)))
    
    return



if __name__ == "__main__":
    all_folds = ['index1.mat', 'index2.mat', 'index3.mat', 'index4.mat', 'index5.mat',
                'index6.mat', 'index7.mat', 'index8.mat', 'index9.mat', 'index10.mat']
    
    if args.smoke_test:
        all_folds = ['index1.mat']
    num_fold = len(all_folds)

    alpha_list = adjust_alpha(args.epochs)
    data_path = os.path.join(args.data_path, args.ds)
    index_path = os.path.join(data_path, args.index)
    mat_name = args.ds + "_" + args.ds_suffix + ".mat"
    logging.info('MAT File Name: {}'.format(mat_name))
    mat_path = os.path.join(data_path, mat_name)
    ds_name = mat_name[0:-4]
    all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab, partial_bag_lab_processed = load_data_mat(
        mat_path, args.nr_fea, args.nr_class, normalize=args.normalize)

    num_trial = 1
    time_s = time.time()

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        logging.info('\tGPU is available!')
        loader_kwargs = {'num_workers': 1, 'pin_memory': True} 
    else: 
        loader_kwargs = {}
    
    logging.info("\t================================ START ========================================")
    
    run()

    logging.info("\t================================ END ========================================")

    time_e = time.time()
    logging.info('\tRunning time is {} seconds.'.format(time_e - time_s))
    logging.info('Training is finished.')
    