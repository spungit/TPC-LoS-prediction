import os

import math
import json

import numpy as np
import pandas as pd
from sklearn import metrics
from itertools import groupby, islice

import torch
import torch.nn as nn
from torch import cat, exp
import torch.nn.functional as F
from torch.nn.functional import pad
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import Adam

from eICU_preprocessing.split_train_test import process_table, shuffle_stays

###################################### MODELS ######################################
## TPC MODEL

# Mean Squared Logarithmic Error (MSLE) loss
class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()
        self.squared_error = nn.MSELoss(reduction='none')

    def forward(self, y_hat, y, mask, seq_length, sum_losses=False):
        # the log(predictions) corresponding to no data should be set to 0
        log_y_hat = y_hat.log().where(mask, torch.zeros_like(y))
        # the we set the log(labels) that correspond to no data to be 0 as well
        log_y = y.log().where(mask, torch.zeros_like(y))
        # where there is no data log_y_hat = log_y = 0, so the squared error will be 0 in these places
        loss = self.squared_error(log_y_hat, log_y)
        loss = torch.sum(loss, dim=1)
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1)
        return loss.mean()

# Mean Squared Error (MSE) loss
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.squared_error = nn.MSELoss(reduction='none')

    def forward(self, y_hat, y, mask, seq_length, sum_losses=False):
        # the predictions corresponding to no data should be set to 0
        y_hat = y_hat.where(mask, torch.zeros_like(y))
        # the we set the labels that correspond to no data to be 0 as well
        y = y.where(mask, torch.zeros_like(y))
        # where there is no data log_y_hat = log_y = 0, so the squared error will be 0 in these places
        loss = self.squared_error(y_hat, y)
        loss = torch.sum(loss, dim=1)
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1)
        return loss.mean()


class MyBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MyBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        # hack to work around model.eval() issue
        if not self.training:
            self.eval_momentum = 0  # set the momentum to zero when the model is validating

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum if self.training else self.eval_momentum

        if self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum if self.training else self.eval_momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            training=True, momentum=exponential_average_factor, eps=self.eps)  # set training to True so it calculates the norm of the batch


class MyBatchNorm1d(MyBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class EmptyModule(nn.Module):
    def forward(self, X):
        return X


class TempPointConv(nn.Module):
    def __init__(self, config, F=None, D=None, no_flat_features=None):

        # The timeseries data will be of dimensions B * (2F + 2) * T where:
        #   B is the batch size
        #   F is the number of features for convolution (N.B. we start with 2F because there are corresponding mask features)
        #   T is the number of timepoints
        #   The other 2 features represent the sequence number and the hour in the day

        # The diagnoses data will be of dimensions B * D where:
        #   D is the number of diagnoses
        # The flat data will be of dimensions B * no_flat_features

        super(TempPointConv, self).__init__()
        self.task = config.task
        self.n_layers = config.n_layers
        self.model_type = config.model_type
        self.share_weights = config.share_weights
        self.diagnosis_size = config.diagnosis_size
        self.main_dropout_rate = config.main_dropout_rate
        self.temp_dropout_rate = config.temp_dropout_rate
        self.kernel_size = config.kernel_size
        self.temp_kernels = config.temp_kernels
        self.point_sizes = config.point_sizes
        self.batchnorm = config.batchnorm
        self.last_linear_size = config.last_linear_size
        self.F = F
        self.D = D
        self.no_flat_features = no_flat_features
        self.no_diag = config.no_diag
        self.no_mask = config.no_mask
        self.no_exp = config.no_exp
        self.no_skip_connections = config.no_skip_connections
        self.alpha = config.alpha
        self.momentum = 0.01 if self.batchnorm == 'low_momentum' else 0.1

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.hardtanh = nn.Hardtanh(min_val=1/48, max_val=100)  # keep the end predictions between half an hour and 100 days
        self.msle_loss = MSLELoss()
        self.mse_loss = MSELoss()
        self.bce_loss = nn.BCELoss()

        self.main_dropout = nn.Dropout(p=self.main_dropout_rate)
        self.temp_dropout = nn.Dropout(p=self.temp_dropout_rate)

        self.remove_none = lambda x: tuple(xi for xi in x if xi is not None)  # removes None items from a tuple
        self.empty_module = EmptyModule()

        if self.batchnorm in ['mybatchnorm', 'pointonly', 'temponly', 'low_momentum']:
            self.batchnormclass = MyBatchNorm1d
        elif self.batchnorm == 'default':
            self.batchnormclass = nn.BatchNorm1d

        # input shape: B * D
        # output shape: B * diagnosis_size
        self.diagnosis_encoder = nn.Linear(in_features=self.D, out_features=self.diagnosis_size)

        if self.batchnorm in ['mybatchnorm', 'pointonly', 'low_momentum', 'default']:
            self.bn_diagnosis_encoder = self.batchnormclass(num_features=self.diagnosis_size, momentum=self.momentum)  # input shape: B * diagnosis_size
            self.bn_point_last_los = self.batchnormclass(num_features=self.last_linear_size, momentum=self.momentum)  # input shape: (B * T) * last_linear_size
            self.bn_point_last_mort = self.batchnormclass(num_features=self.last_linear_size, momentum=self.momentum)  # input shape: (B * T) * last_linear_size
        else:
            self.bn_diagnosis_encoder = self.empty_module
            self.bn_point_last_los = self.empty_module
            self.bn_point_last_mort = self.empty_module

        # input shape: (B * T) * last_linear_size
        # output shape: (B * T) * 1
        self.point_final_los = nn.Linear(in_features=self.last_linear_size, out_features=1)
        self.point_final_mort = nn.Linear(in_features=self.last_linear_size, out_features=1)

        if self.model_type == 'tpc':
            self.init_tpc()
        elif self.model_type == 'temp_only':
            self.init_temp()
        elif self.model_type == 'pointwise_only':
            self.init_pointwise()
        else:
            raise NotImplementedError('Specified model type not supported; supported types include tpc, temp_only and pointwise_only')


    def init_tpc(self):

        # non-module layer attributes
        self.layers = []
        for i in range(self.n_layers):
            dilation = i * (self.kernel_size - 1) if i > 0 else 1  # dilation = 1 for the first layer, after that it captures all the information gathered by previous layers
            temp_k = self.temp_kernels[i]
            point_size = self.point_sizes[i]
            self.update_layer_info(layer=i, temp_k=temp_k, point_size=point_size, dilation=dilation, stride=1)

        # module layer attributes
        self.create_temp_pointwise_layers()

        # input shape: (B * T) * ((F + Zt) * (1 + Y) + diagnosis_size + no_flat_features)
        # output shape: (B * T) * last_linear_size
        input_size = (self.F + self.Zt) * (1 + self.Y) + self.diagnosis_size + self.no_flat_features
        if self.no_diag:
            input_size = input_size - self.diagnosis_size
        if self.no_skip_connections:
            input_size = self.F * self.Y + self.Z + self.diagnosis_size + self.no_flat_features
        self.point_last_los = nn.Linear(in_features=input_size, out_features=self.last_linear_size)
        self.point_last_mort = nn.Linear(in_features=input_size, out_features=self.last_linear_size)

        return


    def init_temp(self):

        # non-module layer attributes
        self.layers = []
        for i in range(self.n_layers):
            dilation = i * (self.kernel_size - 1) if i > 0 else 1  # dilation = 1 for the first layer, after that it captures all the information gathered by previous layers
            temp_k = self.temp_kernels[i]
            self.update_layer_info(layer=i, temp_k=temp_k, dilation=dilation, stride=1)

        # module layer attributes
        self.create_temp_only_layers()

        # input shape: (B * T) * (F * (1 + Y) + diagnosis_size + no_flat_features)
        # output shape: (B * T) * last_linear_size
        input_size = self.F * (1 + self.Y) + self.diagnosis_size + self.no_flat_features
        self.point_last_los = nn.Linear(in_features=input_size, out_features=self.last_linear_size)
        self.point_last_mort = nn.Linear(in_features=input_size, out_features=self.last_linear_size)
        return


    def init_pointwise(self):

        # non-module layer attributes
        self.layers = []
        for i in range(self.n_layers):
            point_size = self.point_sizes[i]
            self.update_layer_info(layer=i, point_size=point_size)

        # module layer attributes
        self.create_pointwise_only_layers()

        # input shape: (B * T) * (Zt + 2F + 2 + no_flat_features + diagnosis_size)
        # output shape: (B * T) * last_linear_size
        if self.no_mask:
            input_size = self.Zt + self.F + 2 + self.no_flat_features + self.diagnosis_size
        else:
            input_size = self.Zt + 2 * self.F + 2 + self.no_flat_features + self.diagnosis_size
        self.point_last_los = nn.Linear(in_features=input_size, out_features=self.last_linear_size)
        self.point_last_mort = nn.Linear(in_features=input_size, out_features=self.last_linear_size)

        return


    def update_layer_info(self, layer=None, temp_k=None, point_size=None, dilation=None, stride=None):

        self.layers.append({})
        if point_size is not None:
            self.layers[layer]['point_size'] = point_size
        if temp_k is not None:
            padding = [(self.kernel_size - 1) * dilation, 0]  # [padding_left, padding_right]
            self.layers[layer]['temp_kernels'] = temp_k
            self.layers[layer]['dilation'] = dilation
            self.layers[layer]['padding'] = padding
            self.layers[layer]['stride'] = stride

        return


    def create_temp_pointwise_layers(self):

        ### Notation used for tracking the tensor shapes ###

        # Z is the number of extra features added by the previous pointwise layer (could be 0 if this is the first layer)
        # Zt is the cumulative number of extra features that have been added by all previous pointwise layers
        # Zt-1 = Zt - Z (cumulative number of extra features minus the most recent pointwise layer)
        # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)

        self.layer_modules = nn.ModuleDict()

        self.Y = 0
        self.Z = 0
        self.Zt = 0

        for i in range(self.n_layers):

            temp_in_channels = (self.F + self.Zt) * (1 + self.Y) if i > 0 else 2 * self.F  # (F + Zt) * (Y + 1)
            temp_out_channels = (self.F + self.Zt) * self.layers[i]['temp_kernels']  # (F + Zt) * temp_kernels
            linear_input_dim = (self.F + self.Zt - self.Z) * self.Y + self.Z + 2 * self.F + 2 + self.no_flat_features  # (F + Zt-1) * Y + Z + 2F + 2 + no_flat_features
            linear_output_dim = self.layers[i]['point_size']  # point_size
            # correct if no_mask
            if self.no_mask:
                if i == 0:
                    temp_in_channels = self.F
                linear_input_dim = (self.F + self.Zt - self.Z) * self.Y + self.Z + self.F + 2 + self.no_flat_features  # (F + Zt-1) * Y + Z + F + 2 + no_flat_features

            temp = nn.Conv1d(in_channels=temp_in_channels,  # (F + Zt) * (Y + 1)
                             out_channels=temp_out_channels,  # (F + Zt) * Y
                             kernel_size=self.kernel_size,
                             stride=self.layers[i]['stride'],
                             dilation=self.layers[i]['dilation'],
                             groups=self.F + self.Zt)

            point = nn.Linear(in_features=linear_input_dim, out_features=linear_output_dim)

            # correct if no_skip_connections
            if self.no_skip_connections:
                temp_in_channels = self.F * self.Y if i > 0 else 2 * self.F  # F * Y
                temp_out_channels = self.F * self.layers[i]['temp_kernels']  # F * temp_kernels
                #linear_input_dim = self.F * self.Y + self.Z if i > 0 else 2 * self.F + 2 + self.no_flat_features  # (F * Y) + Z
                linear_input_dim = self.Z if i > 0 else 2 * self.F + 2 + self.no_flat_features  # Z
                temp = nn.Conv1d(in_channels=temp_in_channels,
                                 out_channels=temp_out_channels,
                                 kernel_size=self.kernel_size,
                                 stride=self.layers[i]['stride'],
                                 dilation=self.layers[i]['dilation'],
                                 groups=self.F)

                point = nn.Linear(in_features=linear_input_dim, out_features=linear_output_dim)

            if self.batchnorm in ['default', 'mybatchnorm', 'low_momentum']:
                bn_temp = self.batchnormclass(num_features=temp_out_channels, momentum=self.momentum)
                bn_point = self.batchnormclass(num_features=linear_output_dim, momentum=self.momentum)
            elif self.batchnorm == 'temponly':
                bn_temp = self.batchnormclass(num_features=temp_out_channels)
                bn_point = self.empty_module
            elif self.batchnorm == 'pointonly':
                bn_temp = self.empty_module
                bn_point = self.batchnormclass(num_features=linear_output_dim)
            else:
                bn_temp = bn_point = self.empty_module  # linear module; does nothing

            self.layer_modules[str(i)] = nn.ModuleDict({
                'temp': temp,
                'bn_temp': bn_temp,
                'point': point,
                'bn_point': bn_point})

            self.Y = self.layers[i]['temp_kernels']
            self.Z = linear_output_dim
            self.Zt += self.Z

        return


    def create_temp_only_layers(self):

        # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)
        self.layer_modules = nn.ModuleDict()
        self.Y = 0

        for i in range(self.n_layers):

            if self.share_weights:
                temp_in_channels = (1 + self.Y) if i > 0 else 2  # (Y + 1)
                temp_out_channels = self.layers[i]['temp_kernels']
                groups = 1
            else:
                temp_in_channels = self.F * (1 + self.Y) if i > 0 else 2 * self.F  # F * (Y + 1)
                temp_out_channels = self.F * self.layers[i]['temp_kernels']  # F * temp_kernels
                groups = self.F

            temp = nn.Conv1d(in_channels=temp_in_channels,
                             out_channels=temp_out_channels,
                             kernel_size=self.kernel_size,
                             stride=self.layers[i]['stride'],
                             dilation=self.layers[i]['dilation'],
                             groups=groups)

            if self.batchnorm in ['default', 'mybatchnorm', 'low_momentum', 'temponly']:
                bn_temp = self.batchnormclass(num_features=temp_out_channels, momentum=self.momentum)
            else:
                bn_temp = self.empty_module  # linear module; does nothing

            self.layer_modules[str(i)] = nn.ModuleDict({
                'temp': temp,
                'bn_temp': bn_temp})

            self.Y = self.layers[i]['temp_kernels']

        return


    def create_pointwise_only_layers(self):

        # Zt is the cumulative number of extra features that have been added by previous pointwise layers
        self.layer_modules = nn.ModuleDict()
        self.Zt = 0

        for i in range(self.n_layers):

            linear_input_dim = self.Zt + 2 * self.F + 2 + self.no_flat_features  # Zt + 2F + 2 + no_flat_features
            linear_output_dim = self.layers[i]['point_size']  # point_size

            if self.no_mask:
                linear_input_dim = self.Zt + self.F + 2 + self.no_flat_features  # Zt + 2F + 2 + no_flat_features

            point = nn.Linear(in_features=linear_input_dim, out_features=linear_output_dim)

            if self.batchnorm in ['default', 'mybatchnorm', 'low_momentum', 'pointonly']:
                bn_point = self.batchnormclass(num_features=linear_output_dim, momentum=self.momentum)
            else:
                bn_point = self.empty_module  # linear module; does nothing

            self.layer_modules[str(i)] = nn.ModuleDict({
                'point': point,
                'bn_point': bn_point})

            self.Zt += linear_output_dim

        return


    # This is really where the crux of TPC is defined. This function defines one TPC layer, as in Figure 3 in the paper:
    # https://arxiv.org/pdf/2007.09483.pdf
    def temp_pointwise(self, B=None, T=None, X=None, repeat_flat=None, X_orig=None, temp=None, bn_temp=None, point=None,
                       bn_point=None, temp_kernels=None, point_size=None, padding=None, prev_temp=None, prev_point=None,
                       point_skip=None):

        ### Notation used for tracking the tensor shapes ###

        # Z is the number of extra features added by the previous pointwise layer (could be 0 if this is the first layer)
        # Zt is the cumulative number of extra features that have been added by all previous pointwise layers
        # Zt-1 = Zt - Z (cumulative number of extra features minus the most recent pointwise layer)
        # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)
        # X shape: B * ((F + Zt) * (Y + 1)) * T; N.B exception in the first layer where there are also mask features, in this case it is B * 2F * T
        # repeat_flat shape: (B * T) * no_flat_features
        # X_orig shape: (B * T) * (2F + 2)
        # prev_temp shape: (B * T) * ((F + Zt-1) * (Y + 1))
        # prev_point shape: (B * T) * Z

        Z = prev_point.shape[1] if prev_point is not None else 0

        X_padded = pad(X, padding, 'constant', 0)  # B * ((F + Zt) * (Y + 1)) * (T + padding)
        X_temp = self.temp_dropout(bn_temp(temp(X_padded)))  # B * ((F + Zt) * temp_kernels) * T

        X_concat = cat(self.remove_none((prev_temp,  # (B * T) * ((F + Zt-1) * Y)
                                         prev_point,  # (B * T) * Z
                                         X_orig,  # (B * T) * (2F + 2)
                                         repeat_flat)),  # (B * T) * no_flat_features
                       dim=1)  # (B * T) * (((F + Zt-1) * Y) + Z + 2F + 2 + no_flat_features)

        point_output = self.main_dropout(bn_point(point(X_concat)))  # (B * T) * point_size

        # point_skip input: B * (F + Zt-1) * T
        # prev_point: B * Z * T
        # point_skip output: B * (F + Zt) * T
        point_skip = cat((point_skip, prev_point.view(B, T, Z).permute(0, 2, 1)), dim=1) if prev_point is not None else point_skip

        temp_skip = cat((point_skip.unsqueeze(2),  # B * (F + Zt) * 1 * T
                         X_temp.view(B, point_skip.shape[1], temp_kernels, T)),  # B * (F + Zt) * temp_kernels * T
                        dim=2)  # B * (F + Zt) * (1 + temp_kernels) * T

        X_point_rep = point_output.view(B, T, point_size, 1).permute(0, 2, 3, 1).repeat(1, 1, (1 + temp_kernels), 1)  # B * point_size * (1 + temp_kernels) * T
        X_combined = self.relu(cat((temp_skip, X_point_rep), dim=1))  # B * (F + Zt) * (1 + temp_kernels) * T
        next_X = X_combined.view(B, (point_skip.shape[1] + point_size) * (1 + temp_kernels), T)  # B * ((F + Zt + point_size) * (1 + temp_kernels)) * T

        temp_output = X_temp.permute(0, 2, 1).contiguous().view(B * T, point_skip.shape[1] * temp_kernels)  # (B * T) * ((F + Zt) * temp_kernels)

        return (temp_output,  # (B * T) * ((F + Zt) * temp_kernels)
                point_output,  # (B * T) * point_size
                next_X,  # B * ((F + Zt) * (1 + temp_kernels)) * T
                point_skip)  # for keeping track of the point skip connections; B * (F + Zt) * T


    def temp(self, B=None, T=None, X=None, X_temp_orig=None, temp=None, bn_temp=None, temp_kernels=None, padding=None):

        ### Notation used for tracking the tensor shapes ###

        # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)
        # X shape: B * (F * (Y + 1)) * T; N.B exception in the first layer where there are also mask features, in this case it is B * 2F * T
        # X_temp_orig shape: B * F * T

        X_padded = pad(X, padding, 'constant', 0)  # B * (F * (Y + 1)) * (T + padding)

        if self.share_weights:
            _, C, padded_length = X_padded.shape
            chans = int(C / self.F)
            X_temp = self.temp_dropout(bn_temp(temp(X_padded.view(B * self.F, chans, padded_length)))).view(B, (self.F * temp_kernels), T)  # B * (F * temp_kernels) * T
        else:
            X_temp = self.temp_dropout(bn_temp(temp(X_padded)))  # B * (F * temp_kernels) * T

        temp_skip = self.relu(cat((X_temp_orig.unsqueeze(2),  # B * F * 1 * T
                                   X_temp.view(B, self.F, temp_kernels, T)),  # B * F * temp_kernels * T
                                   dim=2))  # B * F * (1 + temp_kernels) * T

        next_X = temp_skip.view(B, (self.F * (1 + temp_kernels)), T)  # B * (F * (1 + temp_kernels)) * T

        return next_X  # B * (F * temp_kernels) * T


    def point(self, B=None, T=None, X=None, repeat_flat=None, X_orig=None, point=None, bn_point=None, point_skip=None):

        ### Notation used for tracking the tensor shapes ###

        # Z is the number of extra features added by the previous pointwise layer (could be 0 if this is the first layer)
        # Zt is the cumulative number of extra features that have been added by all previous pointwise layers
        # Zt-1 = Zt - Z (cumulative number of extra features minus the most recent pointwise layer)
        # X shape: B * (F + Zt) * T; N.B exception in the first layer where there are also mask features, in this case it is B * 2F * T
        # repeat_flat shape: (B * T) * no_flat_features
        # X_orig shape: (B * T) * (2F + 2)
        # prev_point shape: (B * T) * Z

        X_combined = cat((X, repeat_flat), dim=1)

        X_point = self.main_dropout(bn_point(point(X_combined)))  # (B * T) * point_size

        # point_skip input: B * Zt-1 * T
        # prev_point: B * Z * T
        # point_skip output: B * Zt * T
        point_skip = cat(self.remove_none((point_skip, X_point.view(B, T, -1).permute(0, 2, 1))), dim=1)

        # point_skip: B * Zt * T
        # X_orig: (B * T) * (2F + 2)
        # repeat_flat: (B * T) * no_flat_features
        # next_X: (B * T) * (Zt + 2F + 2 + no_flat_features)
        next_X = self.relu(cat((point_skip.permute(0, 2, 1).contiguous().view(B * T, -1), X_orig), dim=1))

        return (next_X,  # (B * T) * (Zt + 2F + 2 + no_flat_features)
                point_skip)  # for keeping track of the pointwise skip connections; B * Zt * T


    def temp_pointwise_no_skip(self, B=None, T=None, temp=None, bn_temp=None, point=None, bn_point=None, padding=None, prev_temp=None,
                               prev_point=None, temp_kernels=None, X_orig=None, repeat_flat=None):

        ### Temporal component ###

        # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)
        # prev_temp shape: B * (F * Y) * T; N.B exception in the first layer where there are also mask features, in this case it is B * 2F * T

        X_padded = pad(prev_temp, padding, 'constant', 0)  # B * (F * Y) * (T + padding)
        temp_output = self.relu(self.temp_dropout(bn_temp(temp(X_padded))))  # B * (F * temp_kernels) * T

        ### Pointwise component ###

        # prev_point shape: (B * T) * ((F * Y) + Z)
        point_output = self.relu(self.main_dropout(bn_point(point(prev_point))))  # (B * T) * point_size

        return (temp_output,  # B * (F * temp_kernels) * T
                point_output)  # (B * T) * point_size


    def forward(self, X, diagnoses, flat, time_before_pred=5):

        # flat is B * no_flat_features
        # diagnoses is B * D
        # X is B * (2F + 2) * T
        # X_mask is B * T
        # (the batch is padded to the longest sequence, the + 2 is the time and the hour which are not for temporal convolution)

        # get rid of the time and hour fields - these shouldn't go through the temporal network
        # and split into features and indicator variables
        X_separated = torch.split(X[:, 1:-1, :], self.F, dim=1)  # tuple ((B * F * T), (B * F * T))

        # prepare repeat arguments and initialise layer loop
        B, _, T = X_separated[0].shape
        if self.model_type in ['pointwise_only', 'tpc']:
            repeat_flat = flat.repeat_interleave(T, dim=0)  # (B * T) * no_flat_features
            if self.no_mask:
                X_orig = cat((X_separated[0],
                              X[:, 0, :].unsqueeze(1),
                              X[:, -1, :].unsqueeze(1)), dim=1).permute(0, 2, 1).contiguous().view(B * T, self.F + 2)  # (B * T) * (F + 2)
            else:
                X_orig = X.permute(0, 2, 1).contiguous().view(B * T, 2 * self.F + 2)  # (B * T) * (2F + 2)
            repeat_args = {'repeat_flat': repeat_flat,
                           'X_orig': X_orig,
                           'B': B,
                           'T': T}
            if self.model_type == 'tpc':
                if self.no_mask:
                    next_X = X_separated[0]
                else:
                    next_X = torch.stack(X_separated, dim=2).reshape(B, 2 * self.F, T)  # B * 2F * T
                point_skip = X_separated[0]  # keeps track of skip connections generated from linear layers; B * F * T
                temp_output = None
                point_output = None
            else:  # pointwise only
                next_X = X_orig
                point_skip = None
        elif self.model_type == 'temp_only':
            next_X = torch.stack(X_separated, dim=2).view(B, 2 * self.F, T)  # B * 2F * T
            X_temp_orig = X_separated[0]  # skip connections for temp only model
            repeat_args = {'X_temp_orig': X_temp_orig,
                           'B': B,
                           'T': T}

        if self.no_skip_connections:
            temp_output = next_X
            point_output = cat((X_orig,  # (B * T) * (2F + 2)
                                repeat_flat),  # (B * T) * no_flat_features
                               dim=1)  # (B * T) * (2F + 2 + no_flat_features)
            self.layer1 = True

        for i in range(self.n_layers):
            kwargs = dict(self.layer_modules[str(i)], **repeat_args)
            if self.model_type == 'tpc':
                if self.no_skip_connections:
                    temp_output, point_output = self.temp_pointwise_no_skip(prev_point=point_output, prev_temp=temp_output,
                                                                            temp_kernels=self.layers[i]['temp_kernels'],
                                                                            padding=self.layers[i]['padding'], **kwargs)

                else:
                    temp_output, point_output, next_X, point_skip = self.temp_pointwise(X=next_X, point_skip=point_skip,
                                                                        prev_temp=temp_output, prev_point=point_output,
                                                                        temp_kernels=self.layers[i]['temp_kernels'],
                                                                        padding=self.layers[i]['padding'],
                                                                        point_size=self.layers[i]['point_size'],
                                                                        **kwargs)
            elif self.model_type == 'temp_only':
                next_X = self.temp(X=next_X, temp_kernels=self.layers[i]['temp_kernels'],
                                   padding=self.layers[i]['padding'], **kwargs)
            elif self.model_type == 'pointwise_only':
                next_X, point_skip = self.point(X=next_X, point_skip=point_skip, **kwargs)

        # tidy up
        if self.model_type == 'pointwise_only':
            next_X = next_X.view(B, T, -1).permute(0, 2, 1)
        elif self.no_skip_connections:
            # combine the final layer
            next_X = cat((point_output,
                          temp_output.permute(0, 2, 1).contiguous().view(B * T, self.F * self.layers[-1]['temp_kernels'])),
                         dim=1)
            next_X = next_X.view(B, T, -1).permute(0, 2, 1)

        # note that we cut off at time_before_pred hours here because the model is only valid from time_before_pred hours onwards
        if self.no_diag:
            combined_features = cat((flat.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * no_flat_features
                                     next_X[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * (T - time_before_pred), -1)), dim=1)  # (B * (T - time_before_pred)) * (((F + Zt) * (1 + Y)) + no_flat_features) for tpc
        else:
            diagnoses_enc = self.relu(self.main_dropout(self.bn_diagnosis_encoder(self.diagnosis_encoder(diagnoses))))  # B * diagnosis_size
            combined_features = cat((flat.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * no_flat_features
                                     diagnoses_enc.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * diagnosis_size
                                     next_X[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * (T - time_before_pred), -1)), dim=1)  # (B * (T - time_before_pred)) * (((F + Zt) * (1 + Y)) + diagnosis_size + no_flat_features) for tpc

        last_point_los = self.relu(self.main_dropout(self.bn_point_last_los(self.point_last_los(combined_features))))
        last_point_mort = self.relu(self.main_dropout(self.bn_point_last_mort(self.point_last_mort(combined_features))))

        if self.no_exp:
            los_predictions = self.hardtanh(self.point_final_los(last_point_los).view(B, T - time_before_pred))  # B * (T - time_before_pred)
        else:
            los_predictions = self.hardtanh(exp(self.point_final_los(last_point_los).view(B, T - time_before_pred)))  # B * (T - time_before_pred)
        mort_predictions = self.sigmoid(self.point_final_mort(last_point_mort).view(B, T - time_before_pred))  # B * (T - time_before_pred)

        return los_predictions, mort_predictions


    def temp_pointwise_no_skip_old(self, B=None, T=None, temp=None, bn_temp=None, point=None, bn_point=None, padding=None, prev_temp=None,
                               prev_point=None, temp_kernels=None, X_orig=None, repeat_flat=None):

        ### Temporal component ###

        # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)
        # prev_temp shape: B * (F * Y) * T; N.B exception in the first layer where there are also mask features, in this case it is B * 2F * T

        X_padded = pad(prev_temp, padding, 'constant', 0)  # B * (F * Y) * (T + padding)
        temp_output = self.relu(self.temp_dropout(bn_temp(temp(X_padded))))  # B * (F * temp_kernels) * T

        ### Pointwise component ###

        # prev_point shape: (B * T) * ((F * Y) + Z)

        # if this is not layer 1:
        if self.layer1:
            X_concat = prev_point
            self.layer1 = False
        else:
            X_concat = cat((prev_point,
                            prev_temp.permute(0, 2, 1).contiguous().view(B * T, self.F * temp_kernels)),
                           dim=1)

        point_output = self.relu(self.main_dropout(bn_point(point(X_concat))))  # (B * T) * point_size

        return (temp_output,  # B * (F * temp_kernels) * T
                point_output)  # (B * T) * point_size


    def loss(self, y_hat_los, y_hat_mort, y_los, y_mort, mask, seq_lengths, device, sum_losses, loss_type):
        # mort loss
        if self.task == 'mortality':
            loss = self.bce_loss(y_hat_mort, y_mort) * self.alpha
        # los loss
        else:
            bool_type = torch.cuda.BoolTensor if device == torch.device('cuda') else torch.BoolTensor
            if loss_type == 'msle':
                los_loss = self.msle_loss(y_hat_los, y_los, mask.type(bool_type), seq_lengths, sum_losses)
            elif loss_type == 'mse':
                los_loss = self.mse_loss(y_hat_los, y_los, mask.type(bool_type), seq_lengths, sum_losses)
            if self.task == 'LoS':
                loss = los_loss
            # multitask loss
            if self.task == 'multitask':
                loss = los_loss + self.bce_loss(y_hat_mort, y_mort) * self.alpha
        return loss

## LSTM MODEL
class BaseLSTM(nn.Module):
    def __init__(self, config, F=None, D=None, no_flat_features=None):

        # The timeseries data will be of dimensions B * (2F + 2) * T where:
        #   B is the batch size
        #   F is the number of features for convolution (N.B. we start with 2F because there are corresponding mask features)
        #   T is the number of timepoints
        #   The other 2 features represent the sequence number and the hour in the day

        # The diagnoses data will be of dimensions B * D where:
        #   D is the number of diagnoses
        # The flat data will be of dimensions B * no_flat_features

        super(BaseLSTM, self).__init__()
        self.task = config.task
        self.hidden_size = config.hidden_size
        self.bidirectional = config.bidirectional
        self.channelwise = config.channelwise
        self.n_layers = config.n_layers
        self.lstm_dropout_rate = config.lstm_dropout_rate
        self.main_dropout_rate = config.main_dropout_rate
        self.diagnosis_size = config.diagnosis_size
        self.batchnorm = config.batchnorm
        self.last_linear_size = config.last_linear_size
        self.n_layers = config.n_layers
        self.F = F
        self.D = D
        self.no_flat_features = no_flat_features
        self.no_exp = config.no_exp
        self.alpha = config.alpha
        self.momentum = 0.01 if self.batchnorm == 'low_momentum' else 0.1
        self.no_diag = config.no_diag

        self.n_units = self.hidden_size // 2 if self.bidirectional else self.hidden_size
        self.n_dir = 2 if self.bidirectional else 1

        # use the same initialisation as in keras
        for m in self.modules():
            self.init_weights(m)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.hardtanh = nn.Hardtanh(min_val=1 / 48, max_val=100)  # keep the end predictions between half an hour and 100 days
        self.lstm_dropout = nn.Dropout(p=self.lstm_dropout_rate)
        self.main_dropout = nn.Dropout(p=self.main_dropout_rate)
        self.msle_loss = MSLELoss()
        self.mse_loss = MSELoss()
        self.bce_loss = nn.BCELoss()

        self.empty_module = EmptyModule()
        self.remove_none = lambda x: tuple(xi for xi in x if xi is not None)

        if self.channelwise is False:
            # note if it's bidirectional, then we can't assume there's no influence from future timepoints on past ones
            self.lstm = nn.LSTM(input_size=(2*self.F + 2), hidden_size=self.n_units, num_layers=self.n_layers,
                                bidirectional=self.bidirectional, dropout=self.lstm_dropout_rate)
        elif self.channelwise:
            self.channelwise_lstm_list = nn.ModuleList([nn.LSTM(input_size=2, hidden_size=self.n_units,
                                                                num_layers=self.n_layers, bidirectional=self.bidirectional,
                                                                dropout=self.lstm_dropout_rate) for i in range(self.F)])

        # input shape: B * D
        # output shape: B * diagnosis_size
        self.diagnosis_encoder = nn.Linear(in_features=self.D, out_features=self.diagnosis_size)

        # input shape: B * diagnosis_size
        if self.batchnorm in ['mybatchnorm', 'low_momentum']:
            self.bn_diagnosis_encoder = MyBatchNorm1d(num_features=self.diagnosis_size, momentum=self.momentum)
        elif self.batchnorm == 'default':
            self.bn_diagnosis_encoder = nn.BatchNorm1d(num_features=self.diagnosis_size)
        else:
            self.bn_diagnosis_encoder = self.empty_module

        # input shape: (B * T) * (n_units + diagnosis_size + no_flat_features)
        # output shape: (B * T) * last_linear_size
        channel_wise = self.F if self.channelwise else 1
        input_size = self.n_units * channel_wise + self.diagnosis_size + self.no_flat_features
        if self.no_diag:
            input_size = input_size - self.diagnosis_size
        self.point_los = nn.Linear(in_features=input_size, out_features=self.last_linear_size)
        self.point_mort = nn.Linear(in_features=input_size, out_features=self.last_linear_size)

        # input shape: (B * T) * last_linear_size
        if self.batchnorm in ['mybatchnorm', 'pointonly', 'low_momentum']:
            self.bn_point_last_los = MyBatchNorm1d(num_features=self.last_linear_size, momentum=self.momentum)
            self.bn_point_last_mort = MyBatchNorm1d(num_features=self.last_linear_size, momentum=self.momentum)
        elif self.batchnorm == 'default':
            self.bn_point_last_los = nn.BatchNorm1d(num_features=self.last_linear_size)
            self.bn_point_last_mort = nn.BatchNorm1d(num_features=self.last_linear_size)
        else:
            self.bn_point_last_los = self.empty_module
            self.bn_point_last_mort = self.empty_module

        # input shape: (B * T) * last_linear_size
        # output shape: (B * T) * 1
        self.point_final_los = nn.Linear(in_features=self.last_linear_size, out_features=1)
        self.point_final_mort = nn.Linear(in_features=self.last_linear_size, out_features=1)

        return

    def init_weights(self, m):
        if isinstance(m, nn.LSTM):
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
            for names in m._all_weights:
                for name in filter(lambda n: 'bias' in n, names):
                    bias = getattr(m, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)
        return

    def init_hidden(self, B, device):
        h0 = torch.zeros(self.n_layers * self.n_dir, B, self.n_units).to(device)
        c0 = torch.zeros(self.n_layers * self.n_dir, B, self.n_units).to(device)
        return (h0, c0)

    def forward(self, X, diagnoses, flat, time_before_pred=5):

        # flat is B * no_flat_features
        # diagnoses is B * D
        # X is B * (2F + 2) * T
        # X_mask is B * T
        # (the batch is padded to the longest sequence)

        B, _, T = X.shape
        print('\nX shape:', X.shape)

        if self.channelwise is False:
            # the lstm expects (seq_len, batch, input_size)
            # N.B. the default hidden state is zeros so we don't need to specify it
            lstm_output, hidden = self.lstm(X.permute(2, 0, 1))  # T * B * hidden_size
            print('LSTM output shape:', lstm_output.shape)
            print('Hidden shape:', hidden[0].shape)

        elif self.channelwise is True:
            # take time and hour fields as they are not useful when processed on their own (they go up linearly. They were also taken out for temporal convolution so the comparison is fair)
            X_separated = torch.split(X[:, 1:-1, :], self.F, dim=1)  # tuple ((B * F * T), (B * F * T))
            X_rearranged = torch.stack(X_separated, dim=2)  # B * F * 2 * T
            lstm_output = None
            for i in range(self.F):
                X_lstm, hidden = self.channelwise_lstm_list[i](X_rearranged[:, i, :, :].permute(2, 0, 1))
                lstm_output = cat(self.remove_none((lstm_output, X_lstm)), dim=2)

        X_final = self.relu(self.lstm_dropout(lstm_output.permute(1, 2, 0))) # B * hidden_size * T
        print('X final shape:', X_final.shape)

        # note that we cut off at time_before_pred hours here because the model is only valid from time_before_pred hours onwards
        if self.no_diag:
            combined_features = cat((flat.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * no_flat_features
                                     X_final[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * (T - time_before_pred), -1)), dim=1)
        else:
            diagnoses_enc = self.relu(self.main_dropout(self.bn_diagnosis_encoder(self.diagnosis_encoder(diagnoses))))  # B * diagnosis_size
            combined_features = cat((flat.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * no_flat_features
                                     diagnoses_enc.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * diagnosis_size
                                     X_final[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * (T - time_before_pred), -1)), dim=1)

        last_point_los = self.relu(self.main_dropout(self.bn_point_last_los(self.point_los(combined_features))))
        last_point_mort = self.relu(self.main_dropout(self.bn_point_last_mort(self.point_mort(combined_features))))
        print('Last point mort shape:', last_point_mort.shape)

        if self.no_exp:
            los_predictions = self.hardtanh(self.point_final_los(last_point_los).view(B, T - time_before_pred))  # B * (T - time_before_pred)
        else:
            los_predictions = self.hardtanh(exp(self.point_final_los(last_point_los).view(B, T - time_before_pred)))  # B * (T - time_before_pred)
        mort_predictions = self.sigmoid(self.point_final_mort(last_point_mort).view(B, T - time_before_pred))  # B * (T - time_before_pred)

        return los_predictions, mort_predictions

    def loss(self, y_hat_los, y_hat_mort, y_los, y_mort, mask, seq_lengths, device, sum_losses, loss_type):
        # mort loss
        if self.task == 'mortality':
            loss = self.bce_loss(y_hat_mort, y_mort) * self.alpha
        # los loss
        else:
            bool_type = torch.cuda.BoolTensor if device == torch.device('cuda') else torch.BoolTensor
            if loss_type == 'msle':
                los_loss = self.msle_loss(y_hat_los, y_los, mask.type(bool_type), seq_lengths, sum_losses)
            elif loss_type == 'mse':
                los_loss = self.mse_loss(y_hat_los, y_los, mask.type(bool_type), seq_lengths, sum_losses)
            if self.task == 'LoS':
                loss = los_loss
            # multitask loss
            if self.task == 'multitask':
                loss = los_loss + self.bce_loss(y_hat_mort, y_mort) * self.alpha
        return loss
    
## TRANSFORMER MODEL
# PositionalEncoding adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html. I made the following
# changes:
    # Took out the dropout
    # Changed the dimensions/shape of pe
# I am using the positional encodings suggested by Vaswani et al. as the Attend and Diagnose authors do not specify in
# detail how they do their positional encodings.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=14*24):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0, 2, 1)  # changed from max_len * d_model to 1 * d_model * max_len
        self.register_buffer('pe', pe)

    def forward(self, X):
        # X is B * d_model * T
        # self.pe[:, :, :X.size(2)] is 1 * d_model * T but is broadcast to B when added
        X = X + self.pe[:, :, :X.size(2)]  # B * d_model * T
        return X  # B * d_model * T


class TransformerEncoder(nn.Module):
    def __init__(self, input_size=None, d_model=None, num_layers=None, num_heads=None, feedforward_size=None, dropout=None,
                 pe=None, device=None):
        super(TransformerEncoder, self).__init__()

        self.device = device
        self.d_model = d_model
        self.pe = pe  # boolean variable indicating whether or not the positional encoding should be applied
        self.input_embedding = nn.Conv1d(in_channels=input_size, out_channels=d_model, kernel_size=1)  # B * C * T
        self.pos_encoder = PositionalEncoding(d_model)
        self.trans_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                                              dim_feedforward=feedforward_size, dropout=dropout,
                                                              activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.trans_encoder_layer, num_layers=num_layers)

    def _causal_mask(self, size=None):
        mask = (torch.triu(torch.ones(size, size).to(self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask  # T * T

    def forward(self, X, T):
        # X is B * (2F + 2) * T

        # multiplication by root(d_model) as described in Vaswani et al. 2017 section 3.4
        X = self.input_embedding(X) * math.sqrt(self.d_model)  # B * d_model * T
        if self.pe:  # apply the positional encoding
            X = self.pos_encoder(X)  # B * d_model * T
        X = self.transformer_encoder(src=X.permute(2, 0, 1), mask=self._causal_mask(size=T))  # T * B * d_model
        return X.permute(1, 2, 0)  # B * d_model * T


class Transformer(nn.Module):

    def __init__(self, config, F=None, D=None, no_flat_features=None, device=None):

        # The timeseries data will be of dimensions B * (2F + 2) * T where:
        #   B is the batch size
        #   F is the number of features for convolution (N.B. we start with 2F because there are corresponding mask features)
        #   T is the number of timepoints
        #   The other 2 features represent the sequence number and the hour in the day

        # The diagnoses data will be of dimensions B * D where:
        #   D is the number of diagnoses
        # The flat data will be of dimensions B * no_flat_features

        super(Transformer, self).__init__()
        self.task = config.task
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.feedforward_size = config.feedforward_size
        self.trans_dropout_rate = config.trans_dropout_rate
        self.positional_encoding = config.positional_encoding
        self.main_dropout_rate = config.main_dropout_rate
        self.diagnosis_size = config.diagnosis_size
        self.batchnorm = config.batchnorm
        self.last_linear_size = config.last_linear_size
        self.n_layers = config.n_layers
        self.F = F
        self.D = D
        self.no_flat_features = no_flat_features
        self.no_exp = config.no_exp
        self.alpha = config.alpha
        self.momentum = 0.01 if self.batchnorm == 'low_momentum' else 0.1
        self.no_diag = config.no_diag

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.hardtanh = nn.Hardtanh(min_val=1 / 48, max_val=100)  # keep the end predictions between half an hour and 100 days
        self.trans_dropout = nn.Dropout(p=self.trans_dropout_rate)
        self.main_dropout = nn.Dropout(p=self.main_dropout_rate)
        self.msle_loss = MSLELoss()
        self.mse_loss = MSELoss()
        self.bce_loss = nn.BCELoss()

        self.empty_module = EmptyModule()
        self.remove_none = lambda x: tuple(xi for xi in x if xi is not None)

        self.transformer = TransformerEncoder(input_size=(2*self.F + 2), d_model=self.d_model, num_layers=self.n_layers,
                                              num_heads=self.n_heads, feedforward_size=self.feedforward_size,
                                              dropout=self.trans_dropout_rate, pe=self.positional_encoding,
                                              device=device)

        # input shape: B * D
        # output shape: B * diagnosis_size
        self.diagnosis_encoder = nn.Linear(in_features=self.D, out_features=self.diagnosis_size)

        # input shape: B * diagnosis_size
        if self.batchnorm in ['mybatchnorm', 'low_momentum']:
            self.bn_diagnosis_encoder = MyBatchNorm1d(num_features=self.diagnosis_size, momentum=self.momentum)
        elif self.batchnorm == 'default':
            self.bn_diagnosis_encoder = nn.BatchNorm1d(num_features=self.diagnosis_size)
        else:
            self.bn_diagnosis_encoder = self.empty_module

        # input shape: (B * T) * (d_model + diagnosis_size + no_flat_features)
        # output shape: (B * T) * last_linear_size
        input_size = self.d_model + self.diagnosis_size + self.no_flat_features
        if self.no_diag:
            input_size = input_size - self.diagnosis_size
        self.point_los = nn.Linear(in_features=input_size, out_features=self.last_linear_size)
        self.point_mort = nn.Linear(in_features=input_size, out_features=self.last_linear_size)

        # input shape: (B * T) * last_linear_size
        if self.batchnorm in ['mybatchnorm', 'pointonly', 'low_momentum']:
            self.bn_point_last_los = MyBatchNorm1d(num_features=self.last_linear_size, momentum=self.momentum)
            self.bn_point_last_mort = MyBatchNorm1d(num_features=self.last_linear_size, momentum=self.momentum)
        elif self.batchnorm == 'default':
            self.bn_point_last_los = nn.BatchNorm1d(num_features=self.last_linear_size)
            self.bn_point_last_mort = nn.BatchNorm1d(num_features=self.last_linear_size)
        else:
            self.bn_point_last_los = self.empty_module
            self.bn_point_last_mort = self.empty_module

        # input shape: (B * T) * last_linear_size
        # output shape: (B * T) * 1
        self.point_final_los = nn.Linear(in_features=self.last_linear_size, out_features=1)
        self.point_final_mort = nn.Linear(in_features=self.last_linear_size, out_features=1)

        return

    def forward(self, X, diagnoses, flat, time_before_pred=5):

        # flat is B * no_flat_features
        # diagnoses is B * D
        # X is B * (2F + 2) * T
        # X_mask is B * T
        # (the batch is padded to the longest sequence)

        B, _, T = X.shape  # B * (2F + 2) * T

        trans_output = self.transformer(X, T)  # B * d_model * T

        X_final = self.relu(self.trans_dropout(trans_output))  # B * d_model * T

        # note that we cut off at time_before_pred hours here because the model is only valid from time_before_pred hours onwards
        if self.no_diag:
            combined_features = cat((flat.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * no_flat_features
                                     X_final[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * (T - time_before_pred), -1)), dim=1)
        else:
            diagnoses_enc = self.relu(self.main_dropout(self.bn_diagnosis_encoder(self.diagnosis_encoder(diagnoses))))  # B * diagnosis_size
            combined_features = cat((flat.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * no_flat_features
                                     diagnoses_enc.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * diagnosis_size
                                     X_final[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * (T - time_before_pred), -1)), dim=1)

        last_point_los = self.relu(self.main_dropout(self.bn_point_last_los(self.point_los(combined_features))))
        last_point_mort = self.relu(self.main_dropout(self.bn_point_last_mort(self.point_mort(combined_features))))

        if self.no_exp:
            los_predictions = self.hardtanh(self.point_final_los(last_point_los).view(B, T - time_before_pred))  # B * (T - time_before_pred)
        else:
            los_predictions = self.hardtanh(exp(self.point_final_los(last_point_los).view(B, T - time_before_pred)))  # B * (T - time_before_pred)
        mort_predictions = self.sigmoid(self.point_final_mort(last_point_mort).view(B, T - time_before_pred))  # B * (T - time_before_pred)

        return los_predictions, mort_predictions

    def loss(self, y_hat_los, y_hat_mort, y_los, y_mort, mask, seq_lengths, device, sum_losses, loss_type):
        # mort loss
        if self.task == 'mortality':
            loss = self.bce_loss(y_hat_mort, y_mort) * self.alpha
        # los loss
        else:
            bool_type = torch.cuda.BoolTensor if device == torch.device('cuda') else torch.BoolTensor
            if loss_type == 'msle':
                los_loss = self.msle_loss(y_hat_los, y_los, mask.type(bool_type), seq_lengths, sum_losses)
            elif loss_type == 'mse':
                los_loss = self.mse_loss(y_hat_los, y_los, mask.type(bool_type), seq_lengths, sum_losses)
            if self.task == 'LoS':
                loss = los_loss
            # multitask loss
            if self.task == 'multitask':
                loss = los_loss + self.bce_loss(y_hat_mort, y_mort) * self.alpha
        return loss    

###################################### METRICS ######################################

class CustomBins:
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)

def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0]
        b = CustomBins.bins[i][1]
        if a <= x < b:
            if one_hot:
                onehot = np.zeros((CustomBins.nbins,))
                onehot[i] = 1
                return onehot
            return i
    return None

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(4/24, y_true))) * 100  # this stops the mape being a stupidly large value when y_true happens to be very small

def mean_squared_logarithmic_error(y_true, y_pred):
    return np.mean(np.square(np.log(y_true/y_pred)))

def print_metrics_regression(y_true, predictions, verbose=1):
    print('==> Length of Stay:')
    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]
    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    print('Custom bins confusion matrix:')
    print(cf)

    kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins, weights='linear')
    mad = metrics.mean_absolute_error(y_true, predictions)
    mse = metrics.mean_squared_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)
    msle = mean_squared_logarithmic_error(y_true, predictions)
    r2 = metrics.r2_score(y_true, predictions)

    if verbose:
        print('Mean absolute deviation (MAD) = {}'.format(mad))
        print('Mean squared error (MSE) = {}'.format(mse))
        print('Mean absolute percentage error (MAPE) = {}'.format(mape))
        print('Mean squared logarithmic error (MSLE) = {}'.format(msle))
        print('R^2 Score = {}'.format(r2))
        print('Cohen kappa score = {}'.format(kappa))

    return [mad, mse, mape, msle, r2, kappa]

def print_metrics_mortality(y_true, prediction_probs, verbose=1):
    print('==> Mortality:')
    prediction_probs = np.array(prediction_probs)
    prediction_probs = np.transpose(np.append([1 - prediction_probs], [prediction_probs], axis=0))
    predictions = prediction_probs.argmax(axis=1)
    cf = metrics.confusion_matrix(y_true, predictions, labels=range(2))
    print('N predictions: {}'.format(len(predictions)))
    print('N true: {}'.format(len(y_true)))
    print('Confusion matrix:')
    print(cf)

    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])

    auroc = metrics.roc_auc_score(y_true, prediction_probs[:, 1])
    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, prediction_probs[:, 1])
    auprc = metrics.auc(recalls, precisions)
    f1macro = metrics.f1_score(y_true, predictions, average='macro')

    results = {'Accuracy': acc, 'Precision Survived': prec0, 'Precision Died': prec1, 'Recall Survived': rec0,
               'Recall Died': rec1, 'Area Under the Receiver Operating Characteristic curve (AUROC)': auroc,
               'Area Under the Precision Recall curve (AUPRC)': auprc, 'F1 score (macro averaged)': f1macro}
    if verbose:
        for key in results:
            print('{} = {}'.format(key, results[key]))

    return [acc, prec0, prec1, rec0, rec1, auroc, auprc, f1macro]

###################################### EXPERIMENT TEMPLATE ######################################

def remove_padding(y, mask, device):
    """
        Filters out padding from tensor of predictions or labels

        Args:
            y: tensor of los predictions or labels
            mask (bool_type): tensor showing which values are padding (0) and which are data (1)
    """
    # note it's fine to call .cpu() on a tensor already on the cpu
    y = y.where(mask, torch.tensor(float('nan')).to(device=device)).flatten().detach().cpu().numpy()
    y = y[~np.isnan(y)]
    return y

def shuffle_train(train_path):

    labels = pd.read_csv(train_path + '/labels.csv', index_col='patient')
    flat = pd.read_csv(train_path + '/flat.csv', index_col='patient')
    diagnoses = pd.read_csv(train_path + '/diagnoses.csv', index_col='patient')
    timeseries = pd.read_csv(train_path + '/timeseries.csv', index_col='patient')

    stays = labels.index.values
    stays = shuffle_stays(stays, seed=None)  # No seed will make it completely random
    for table_name, table in zip(['labels', 'flat', 'diagnoses', 'timeseries'],
                                 [labels, flat, diagnoses, timeseries]):
        process_table(table_name, table, stays, train_path)

    with open(train_path + '/stays.txt', 'w') as f:
        for stay in stays:
            f.write("%s\n" % stay)
    return

class ExperimentTemplate():

    def __init__(self, config, model_name):
        self.device = torch.device('cpu')
        # set bool type for where statements
        self.bool_type = torch.cuda.BoolTensor if self.device == torch.device('cuda') else torch.BoolTensor

        # set config
        self.labs_only = config.labs_only
        self.no_labs = config.no_labs
        self.batch_size = config.batch_size
        self.batch_size_test = config.batch_size_test
        self.percentage_data = config.percentage_data
        self.shuffle_train = config.shuffle_train
        self.task = config.task
        self.log_interval = config.log_interval
        self.learning_rate = config.learning_rate
        self.L2_regularisation = config.L2_regularisation
        self.sum_losses = config.sum_losses
        self.loss = config.loss_type

        # get datareader
        self.data_path = config.eICU_path
        self.datareader = eICUReader
        self.train_datareader = self.datareader(self.data_path + 'train', device=self.device,
                                           labs_only=self.labs_only, no_labs=self.no_labs)
        self.val_datareader = self.datareader(self.data_path + 'val', device=self.device,
                                         labs_only=self.labs_only, no_labs=self.no_labs)
        self.test_datareader = self.datareader(self.data_path + 'test', device=self.device,
                                          labs_only=self.labs_only, no_labs=self.no_labs)
        self.no_train_batches = len(self.train_datareader.patients) / self.batch_size

        # set up model and run params
        self.checkpoint_counter = 0

        if model_name == 'LSTM':
            self.model = BaseLSTM(config=config,
                              F=self.train_datareader.F,
                              D=self.train_datareader.D,
                              no_flat_features=self.train_datareader.no_flat_features).to(device=self.device)
            print('Model: LSTM')
            print(self.model)
        elif model_name == 'Transformer':
            self.model = Transformer(config=config,
                            F=self.train_datareader.F,
                            D=self.train_datareader.D,
                            no_flat_features=self.train_datareader.no_flat_features,
                            device=self.device).to(device=self.device)
            print('Model: Transformer')
            print(self.model)
        elif model_name == 'TPC':
            self.model = TempPointConv(config=config,
                                   F=self.train_datareader.F,
                                   D=self.train_datareader.D,
                                   no_flat_features=self.train_datareader.no_flat_features).to(device=self.device)
            print('Model: Temporal Point Convolution')
            print(self.model)

        self.optimiser = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.L2_regularisation)
        self.remove_padding = lambda y, mask: remove_padding(y, mask, device=self.device)

        return

    def train(self, epoch, mort_pred_time=24):
        print('Number of training batches: {}'.format(self.no_train_batches))
        
        self.model.train()
        if epoch > 0 and self.shuffle_train:
            shuffle_train(self.data_path + 'train')  # shuffle the order of the training data to make the batches different, this takes a bit of time
        train_batches = self.train_datareader.batch_gen(batch_size=self.batch_size)
        train_loss = []
        train_y_hat_los = np.array([])
        train_y_los = np.array([])
        train_y_hat_mort = np.array([])
        train_y_mort = np.array([])

        for batch_idx, batch in enumerate(train_batches):

            if len(batch[0]) < 2:
                continue
            else:
                if batch_idx > (self.no_train_batches // (100 / self.percentage_data)):
                    break

                padded, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch
                print('Padded shape:', padded.shape)
                print('Mask shape:', mask.shape)
                print('Diagnoses shape:', diagnoses.shape)
                print('Flat shape:', flat.shape)
                print('LoS labels shape:', los_labels.shape)
                print('Mortality labels shape:', mort_labels.shape)
                print('Sequence lengths shape:', seq_lengths.shape)

                # save sample from batch
                if batch_idx in [0, 1]:
                    padded_sample = padded[0,:,:].detach().cpu().numpy()
                    df = pd.DataFrame(padded_sample)
                    df.to_csv(f'padded_sample_{batch_idx}.csv')

                self.optimiser.zero_grad()
                y_hat_los, y_hat_mort = self.model(padded, diagnoses, flat)
                print('y_hat_los:', y_hat_los.shape)
                print('y_hat_mort:', y_hat_mort.shape)
                loss = self.model.loss(y_hat_los, y_hat_mort, los_labels, mort_labels, mask, seq_lengths, self.device,
                                    self.sum_losses, self.loss)
                loss.backward()
                self.optimiser.step()
                train_loss.append(loss.item())

                if self.task in ('LoS', 'multitask'):
                    train_y_hat_los = np.append(train_y_hat_los, self.remove_padding(y_hat_los, mask.type(self.bool_type)))
                    train_y_los = np.append(train_y_los, self.remove_padding(los_labels, mask.type(self.bool_type)))
                if self.task in ('mortality', 'multitask') and mort_labels.shape[1] >= mort_pred_time:
                    train_y_hat_mort = np.append(train_y_hat_mort,
                                                self.remove_padding(y_hat_mort[:, mort_pred_time],
                                                                    mask.type(self.bool_type)[:, mort_pred_time]))
                    train_y_mort = np.append(train_y_mort, self.remove_padding(mort_labels[:, mort_pred_time],
                                                                            mask.type(self.bool_type)[:, mort_pred_time]))

                mean_loss_report = sum(train_loss[(batch_idx - self.log_interval):-1]) / self.log_interval
                print('Epoch: {} [{:5.0f}/{:5.0f} samples] | train loss: {:3.4f}'.format(epoch,
                                                                            batch_idx * self.batch_size,
                                                                            batch_idx * self.no_train_batches,
                                                                            mean_loss_report))

        print('Train Metrics:')
        mean_train_loss = sum(train_loss) / len(train_loss)
        if self.task in ('LoS', 'multitask'):
            print_metrics_regression(train_y_los, train_y_hat_los) # order: mad, mse, mape, msle, r2, kappa
        if self.task in ('mortality', 'multitask'):
            print_metrics_mortality(train_y_mort, train_y_hat_mort)
        print('Epoch: {} | Train Loss: {:3.4f}'.format(epoch, mean_train_loss))

        return

    def validate(self, epoch, mort_pred_time=24):

        self.model.eval()
        val_batches = self.val_datareader.batch_gen(batch_size=self.batch_size_test)
        val_loss = []
        val_y_hat_los = np.array([])
        val_y_los = np.array([])
        val_y_hat_mort = np.array([])
        val_y_mort = np.array([])

        for batch in val_batches:

            if batch[0].shape[0] < 2:
                continue
            else:
                padded, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch

                y_hat_los, y_hat_mort = self.model(padded, diagnoses, flat)
                loss = self.model.loss(y_hat_los, y_hat_mort, los_labels, mort_labels, mask, seq_lengths, self.device,
                                        self.sum_losses, self.loss)
                val_loss.append(loss.item())  # can't add the model.loss directly because it causes a memory leak

                if self.task in ('LoS', 'multitask'):
                    val_y_hat_los = np.append(val_y_hat_los,
                                                self.remove_padding(y_hat_los, mask.type(self.bool_type)))
                    val_y_los = np.append(val_y_los, self.remove_padding(los_labels, mask.type(self.bool_type)))
                if self.task in ('mortality', 'multitask') and mort_labels.shape[1] >= mort_pred_time:
                    val_y_hat_mort = np.append(val_y_hat_mort,
                                                    self.remove_padding(y_hat_mort[:, mort_pred_time],
                                                                        mask.type(self.bool_type)[:, mort_pred_time]))
                    val_y_mort = np.append(val_y_mort, self.remove_padding(mort_labels[:, mort_pred_time],
                                                                            mask.type(self.bool_type)[:, mort_pred_time]))

        print('Validation Metrics:')
        mean_val_loss = sum(val_loss) / len(val_loss)
        if self.task in ('LoS', 'multitask'):
            print_metrics_regression(val_y_los, val_y_hat_los) # order: mad, mse, mape, msle, r2, kappa
        if self.task in ('mortality', 'multitask'):
            print_metrics_mortality(val_y_mort, val_y_hat_mort)
        print('Epoch: {} | Validation Loss: {:3.4f}'.format(epoch, mean_val_loss))

        return

    def test(self, mort_pred_time=24):

        self.model.eval()
        test_batches = self.test_datareader.batch_gen(batch_size=self.batch_size_test)
        test_loss = []
        test_y_hat_los = np.array([])
        test_y_los = np.array([])
        test_y_hat_mort = np.array([])
        test_y_mort = np.array([])

        for batch in test_batches:

            if batch[0].shape[0] < 2:
                continue
            else:
                padded, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch
                print('Padded shape:', padded.shape)
                print('Mask shape:', mask.shape)
                print('Diagnoses shape:', diagnoses.shape)
                print('Flat shape:', flat.shape)
                print('LoS labels shape:', los_labels.shape)
                print('Mortality labels shape:', mort_labels.shape)
                print('Sequence lengths shape:', seq_lengths.shape)

                y_hat_los, y_hat_mort = self.model(padded, diagnoses, flat)
                loss = self.model.loss(y_hat_los, y_hat_mort, los_labels, mort_labels, mask, seq_lengths, self.device,
                                    self.sum_losses, self.loss)
                test_loss.append(loss.item())  # can't add the model.loss directly because it causes a memory leak

                if self.task in ('LoS', 'multitask'):
                    test_y_hat_los = np.append(test_y_hat_los,
                                            self.remove_padding(y_hat_los, mask.type(self.bool_type)))
                    test_y_los = np.append(test_y_los, self.remove_padding(los_labels, mask.type(self.bool_type)))
                if self.task in ('mortality', 'multitask') and mort_labels.shape[1] >= mort_pred_time:
                    test_y_hat_mort = np.append(test_y_hat_mort,
                                            self.remove_padding(y_hat_mort[:, mort_pred_time],
                                                                mask.type(self.bool_type)[:, mort_pred_time]))
                    test_y_mort = np.append(test_y_mort, self.remove_padding(mort_labels[:, mort_pred_time],
                                                                            mask.type(self.bool_type)[:, mort_pred_time]))

        print('Test Metrics:')
        mean_test_loss = sum(test_loss) / len(test_loss)

        if self.task in ('LoS', 'multitask'):
            print_metrics_regression(test_y_los, test_y_hat_los)  # order: mad, mse, mape, msle, r2, kappa
        if self.task in ('mortality', 'multitask'):
            print_metrics_mortality(test_y_mort, test_y_hat_mort)

        print('Test Loss: {:3.4f}'.format(mean_test_loss))

###################################### DATA LOADER ######################################
# bit hacky but passes checks and I don't have time to implement a neater solution
lab_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 15, 16, 18, 21, 22, 23, 24, 29, 32, 33, 34, 39, 40, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 63, 67, 68, 69, 70, 71, 72, 75, 83, 84, 86]
labs_to_keep = [0] + [(i + 1) for i in lab_indices] + [(i + 88) for i in lab_indices] + [-1]
no_lab_indices = list(range(87))
no_lab_indices = [x for x in no_lab_indices if x not in lab_indices]
no_labs_to_keep = [0] + [(i + 1) for i in no_lab_indices] + [(i + 88) for i in no_lab_indices] + [-1]

class eICUReader(object):

    def __init__(self, data_path, device=None, labs_only=False, no_labs=False):
        self._diagnoses_path = data_path + '/diagnoses.csv'
        self._labels_path = data_path + '/labels.csv'
        self._flat_path = data_path + '/flat.csv'
        self._timeseries_path = data_path + '/timeseries.csv'
        self._device = device
        self.labs_only = labs_only
        self.no_labs = no_labs
        self._dtype = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor

        self.labels = pd.read_csv(self._labels_path, index_col='patient')
        self.flat = pd.read_csv(self._flat_path, index_col='patient')
        self.diagnoses = pd.read_csv(self._diagnoses_path, index_col='patient')

        # we minus 2 to calculate F because hour and time are not features for convolution
        self.F = (pd.read_csv(self._timeseries_path, index_col='patient', nrows=1).shape[1] - 2)//2
        self.D = self.diagnoses.shape[1]
        self.no_flat_features = self.flat.shape[1]

        self.patients = list(self.labels.index)
        self.no_patients = len(self.patients)

    def line_split(self, line):
        return [float(x) for x in line.split(',')]

    def pad_sequences(self, ts_batch):
        seq_lengths = [len(x) for x in ts_batch]
        max_len = max(seq_lengths)
        padded = [patient + [[0] * (self.F * 2 + 2)] * (max_len - len(patient)) for patient in ts_batch]
        if self.labs_only:
            padded = np.array(padded)
            padded = padded[:, :, labs_to_keep]
        if self.no_labs:
            padded = np.array(padded)
            padded = padded[:, :, no_labs_to_keep]
        padded = torch.tensor(padded, device=self._device).type(self._dtype).permute(0, 2, 1)  # B * (2F + 2) * T
        padded[:, 0, :] /= 24  # scale the time into days instead of hours
        mask = torch.zeros(padded[:, 0, :].shape, device=self._device).type(self._dtype)
        for p, l in enumerate(seq_lengths):
            mask[p, :l] = 1
        return padded, mask, torch.tensor(seq_lengths).type(self._dtype)

    def get_los_labels(self, labels, times, mask):
        times = labels.unsqueeze(1).repeat(1, times.shape[1]) - times
        # clamp any labels that are less than 30 mins otherwise it becomes too small when the log is taken
        # make sure where there is no data the label is 0
        return (times.clamp(min=1/48) * mask)

    def get_mort_labels(self, labels, length):
        repeated_labels = labels.unsqueeze(1).repeat(1, length)
        return repeated_labels

    def batch_gen(self, batch_size=8, time_before_pred=5):

        # note that once the generator is finished, the file will be closed automatically
        with open(self._timeseries_path, 'r') as timeseries_file:
            # the first line is the feature names; we have to skip over this
            self.timeseries_header = next(timeseries_file).strip().split(',')
            # this produces a generator that returns a list of batch_size patient identifiers
            patient_batches = (self.patients[pos:pos + batch_size] for pos in range(0, len(self.patients), batch_size))
            # create a generator to capture a single patient timeseries
            ts_patient = groupby(map(self.line_split, timeseries_file), key=lambda line: line[0])
            # we loop through these batches, tracking the index because we need it to index the pandas dataframes
            for i, batch in enumerate(patient_batches):
                ts_batch = [[line[1:] for line in ts] for _, ts in islice(ts_patient, batch_size)]
                padded, mask, seq_lengths = self.pad_sequences(ts_batch)
                los_labels = self.get_los_labels(torch.tensor(self.labels.iloc[i*batch_size:(i+1)*batch_size,7].values, device=self._device).type(self._dtype), padded[:,0,:], mask)
                mort_labels = self.get_mort_labels(torch.tensor(self.labels.iloc[i*batch_size:(i+1)*batch_size,5].values, device=self._device).type(self._dtype), length=mask.shape[1])

                # we must avoid taking data before time_before_pred hours to avoid diagnoses and apache variable from the future
                yield (padded,  # B * (2F + 2) * T
                       mask[:, time_before_pred:],  # B * (T - time_before_pred)
                       torch.tensor(self.diagnoses.iloc[i*batch_size:(i+1)*batch_size].values, device=self._device).type(self._dtype),  # B * D
                       torch.tensor(self.flat.iloc[i*batch_size:(i+1)*batch_size].values.astype(float), device=self._device).type(self._dtype),  # B * no_flat_features
                       los_labels[:, time_before_pred:],
                       mort_labels[:, time_before_pred:],
                       seq_lengths - time_before_pred)

###################################### MAIN + SET PARAMS ######################################
class Configuration:
    def __init__(self, model_name):
        self.task = 'mortality'
        self.loss_type = 'msle'
        self.labs_only = False
        self.no_mask = False
        self.no_diag = False
        self.no_labs = False
        self.no_exp = False
        self.batchnorm = 'mybatchnorm'
        self.shuffle_train = True
        self.percentage_data = 100.0
        self.alpha = 100
        self.main_dropout_rate = 0.45
        self.L2_regularisation = 0
        self.last_linear_size = 17
        self.diagnosis_size = 64
        self.eICU_path = eICU_path
        self.no_diag = False
        self.log_interval = 100
        self.sum_losses = True
        self.batch_size_test = 32

        if model_name == 'LSTM':
            self.n_epochs = 1 #8
            self.batch_size = 512
            self.n_layers = 2
            self.hidden_size = 128
            self.learning_rate = 0.00129
            self.lstm_dropout_rate = 0.2
            self.bidirectional = False
            self.channelwise = False
        elif model_name == 'Transformer':
            self.n_epochs = 15
            self.batch_size = 32
            self.n_layers = 6
            self.feedforward_size = 256
            self.d_model = 16
            self.n_heads = 2
            self.learning_rate = 0.00017
            self.trans_dropout_rate = 0
            self.positional_encoding = True
        elif model_name == 'TPC':
            self.model_type = 'tpc'
            self.n_epochs = 15
            self.batch_size = 32
            self.n_layers = 9
            self.kernel_size = 4
            self.no_temp_kernels = 12
            self.point_size = 13
            self.learning_rate = 0.00226
            self.temp_dropout_rate = 0.05
            self.share_weights = False
            self.no_skip_connections = False
            self.temp_kernels = [self.no_temp_kernels]*self.n_layers
            self.point_sizes = [self.point_size]*self.n_layers

if __name__ == '__main__':

    with open('paths.json', 'r') as f:
        eICU_path = json.load(f)["eICU_path"]

    # RUN LSTM
    config = Configuration(model_name='LSTM')
    lstm_experiment = ExperimentTemplate(config=config, model_name='LSTM')

    for epoch in range(config.n_epochs):
        lstm_experiment.train(epoch)
        lstm_experiment.validate(epoch)
        lstm_experiment.test()

    # ## RUN TRANSFORMER
    # config = Configuration(model_name='Transformer')
    # transformer_experiment = ExperimentTemplate(config=config, model_name='Transformer')

    # for epoch in range(config.n_epochs):
    #     transformer_experiment.train(epoch)
    #     transformer_experiment.validate(epoch)
    #     transformer_experiment.test()

    # ## RUN TPC
    # config = Configuration(model_name='TPC')    
    # tpc_experiment = ExperimentTemplate(config=config, model_name='TPC')

    # for epoch in range(config.n_epochs):
    #     tpc_experiment.train(epoch)
    #     tpc_experiment.validate(epoch)
    #     tpc_experiment.test()