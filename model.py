#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


from utils import device, clamp_probs
from utils_nn import GatedAttentionLayerV, GatedAttentionLayerU
from utils_nn import MLP, mipl_sigmoid




class VAEAttention(nn.Module):
    def __init__(self, args):
        super(VAEAttention, self).__init__()
        self.ds_name = args.ds
        if self.ds_name in ["CRC_MIPL"]:
            self.L = 512
        else:
            self.L = 128
        self.attention_dim = args.attention_dim
        self.K = 1
        self.nr_fea = args.nr_fea
        self.nr_class = args.nr_class

        self.tau = args.init_tau
        self.decay_tau = args.decay_tau
        self.min_tau = args.min_tau

        self.clamp_flag = args.clamp_flag
        self.w_ent = args.w_ent
        self.w_rec = args.w_rec

        if 'mnist' in self.ds_name.lower():
            self.feature_extractor_part1 = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            )
            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(50 * 4 * 4, self.L),
                nn.Dropout(),
                nn.ReLU()
            )
        else:   # for Birdsong_MIPL, SIVAL_MIPL, CRC-MIPL datasets
            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(self.nr_fea, self.L),
                nn.Dropout(), 
                nn.ReLU()
            )
        self.att_layer_v = GatedAttentionLayerV(self.L)
        self.att_layer_u = GatedAttentionLayerU(self.L)
        self.linear_v = nn.Linear(self.L * self.K, self.nr_class)
        self.linear_u = nn.Linear(self.L * self.K, self.nr_class)
        self.attention_weights = nn.Sequential(
            nn.Linear(self.nr_class, self.attention_dim),
            nn.ReLU(),
            nn.Linear(self.attention_dim, self.K)
        ) 

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.nr_class),
            nn.Sigmoid()
        )

        ## Disambiguation
        # Inference model
        self.inference_model = MLP(self.L * self.K + self.nr_class, 256, [256, 512], False,
                           active_func="relu")
        self.inference_model_fc = nn.Linear(256, self.nr_class)
        # Generative model
        self.generate_model = MLP(self.L * self.K + self.nr_class, 512, [256], False,
                             active_func="relu")
        self.generate_model_fc = nn.Linear(512, self.nr_class)

    def _inf(self, input, candidate_labels):
        result = self.inference_model(torch.cat([input, candidate_labels.unsqueeze(0)], dim=1))
        result = torch.sigmoid(self.inference_model_fc(result)) * candidate_labels
        
        return result
    
    def _gen(self, input, truth_labels):
        result = self.generate_model(torch.cat([input, truth_labels], dim=1))
        result = torch.sigmoid(self.generate_model_fc(result))
        
        return result

    def decay_temperature(self):
        self.tau *= (1. - self.decay_tau)
        self.tau = max(self.tau, self.min_tau)

    def forward(self, x):
        x = x.squeeze(0)


        if 'mnist' in self.ds_name.lower():
            x = x.reshape(x.shape[0], 1, 28, 28)
            h = self.feature_extractor_part1(x)
            h = h.view(-1, 50 * 4 * 4)
            h = self.feature_extractor_part2(h)
        else:   # for Birdsong_MIPL, SIVAL_MIPL, CRC-MIPL datasets
            x = x.float()
            h = self.feature_extractor_part2(x)  
        a_v = self.att_layer_v(h, self.linear_v.weight, self.linear_v.bias)
        a_u = self.att_layer_u(h, self.linear_v.weight, self.linear_v.bias)
        a = self.attention_weights(a_v * a_u)
        a = torch.transpose(a, 1, 0)
        # Equation (4):
        a = (a - a.mean()) / (torch.std(a.detach()) + 1e-8)
        a = a / math.sqrt(self.L)
        a = F.softmax(a / self.tau, dim=1)
        z = torch.mm(a, h) 
        y_logits = self.classifier(z)

        return y_logits, z
    
    def full_loss(self, prediction, bag_y_can, zb):
        # q(y|z,s)
        truth_conv = self._inf(zb, bag_y_can)
        # Sampling with mipl_sigmoid
        truth_targets = mipl_sigmoid(truth_conv)
        # p(s|z,y)
        rec_conv = self._gen(zb, truth_targets)
        # Class Loss $\mathcal{L}_{P R I}(\theta, \mathcal{D})$
        entropy_cls = - truth_conv.detach() * torch.log(prediction)
        cls_loss = (torch.sum(entropy_cls)) / entropy_cls.size(0)
        # Entropy loss $\mathcal{L}_{CE}(\theta, \phi, \mathcal{D})$
        truth_conv = clamp_probs(truth_conv)
        ent_loss = -torch.mean(truth_conv * torch.log(truth_conv) + (1-truth_conv) * torch.log1p(-truth_conv))
        # Reconstruction loss $\mathcal{L}_{K L}(\theta, \phi, \mathcal{D})$
        rec_loss = F.cross_entropy(rec_conv, bag_y_can.unsqueeze(0))
        
        ## probability disambiguation mechanism
        # Total loss
        loss = cls_loss - self.w_ent * ent_loss + self.w_rec * rec_loss
        y_candiate = torch.zeros(bag_y_can.shape).to(device)
        y_candiate[bag_y_can > 0] = 1
        prediction_can = rec_conv * y_candiate
        new_prediction = prediction_can / prediction_can.sum(dim=1).repeat(prediction_can.size(1), 1).transpose(0, 1)

        return new_prediction, loss

    def calculate_objective(self, x, bag_y_can):
        '''
        calculate the full loss
        '''
        bag_y_can = bag_y_can.reshape(-1)
        y_logits, z = self.forward(x)
        if self.clamp_flag:
            y_logits = torch.clamp(y_logits, min=1e-5, max=1.-1e-5)
        y_prob = F.softmax(y_logits, dim=1)
        new_prob, loss = self.full_loss(y_prob, bag_y_can, z)

        return loss, new_prob


    def evaluate_objective(self, x):
        '''
        model testing
        '''
        y_prob, _ = self.forward(x)
        y_prob = F.softmax(y_prob, dim=1)

        return y_prob
