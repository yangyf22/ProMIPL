import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import clamp_probs


class GatedAttentionLayerV(nn.Module):
    '''
    $\text{tanh}\left(\boldsymbol{W}_{v}^\top \boldsymbol{h}_{i,j} + \boldsymbol{b}_v \right)$ in Equation (2)
    '''
    def __init__(self, dim=512):
        super(GatedAttentionLayerV, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)


    def forward(self, features, W_V, b_V):
        out = F.linear(features, W_V, b_V)
        out_tanh = torch.tanh(out)

        return out_tanh


class GatedAttentionLayerU(nn.Module):
    '''
    $\text{sigm}\left(\boldsymbol{W}_{u}^\top \boldsymbol{h}_{i,j} + \boldsymbol{b}_u \right)$ in Equation (2)
    '''
    def __init__(self, dim=512):
        super(GatedAttentionLayerU, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)


    def forward(self, features, W_U, b_U):
        out = F.linear(features, W_U, b_U)
        out_sigmoid = torch.sigmoid(out)

        return out_sigmoid


class MLP(nn.Module):
    def __init__(self, in_fea, out_fea, hidden_fea=[], batchNorm=False,
                 drop_ratio=0.0, active_func='relu', neg_slope=0.1,
                 with_output_active_func=True):
        super(MLP, self).__init__()
        self.nonlinearity = active_func
        self.negative_slope = neg_slope
        self.fcs = nn.ModuleList()
        if hidden_fea:
            in_dims = [in_fea] + hidden_fea
            out_dims = hidden_fea + [out_fea]
            for i in range(len(in_dims)):
                self.fcs.append(nn.Linear(in_dims[i], out_dims[i]))
                if with_output_active_func or i < len(hidden_fea):
                    if batchNorm:
                        self.fcs.append(nn.BatchNorm1d(out_dims[i], track_running_stats=True))
                    if active_func == 'relu':
                        self.fcs.append(nn.ReLU(inplace=True))
                    else:
                        raise ValueError("Unsupported nonlinearity {}".format(active_func))
                    if drop_ratio:
                        self.fcs.append(nn.Dropout(drop_ratio))
        else:
            self.fcs.append(nn.Linear(in_fea, out_fea))
            if with_output_active_func:
                if batchNorm:
                    self.fcs.append(nn.BatchNorm1d(out_fea, track_running_stats=True))
                if active_func == 'relu':
                    self.fcs.append(nn.ReLU(inplace=True))
                else:
                    raise ValueError("Unsupported nonlinearity {}".format(active_func))
                if drop_ratio:
                        self.fcs.append(nn.Dropout(drop_ratio))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for l in self.fcs:
            if l.__class__.__name__ == 'Linear':
                nn.init.kaiming_uniform_(l.weight, a=self.negative_slope,
                                         nonlinearity=self.nonlinearity)
                if self.nonlinearity == 'relu':
                    nn.init.uniform_(l.bias, 0, 0.1)
                else:
                    nn.init.constant_(l.bias, 0.0)
            elif l.__class__.__name__ == 'BatchNorm1d':
                l.reset_parameters()
    
    def forward(self, input):
        for l in self.fcs:
            input = l(input)
        return input

def mipl_sigmoid(probs):
    probs = clamp_probs(probs)
    logits = torch.log(probs) - torch.log1p(-probs)
    uniforms = clamp_probs(torch.rand(logits.size(), device=logits.device, dtype=logits.dtype))
    samples = uniforms.log() - (-uniforms).log1p() + logits
    y_soft = torch.sigmoid(samples * 1.5)
    ret = y_soft
    return ret

