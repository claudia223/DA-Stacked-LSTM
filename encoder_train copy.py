import typing
from typing import Tuple
import json
import os

import torch
import joblib
from torch import nn
from torch import optim
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utils
from modules import *
from main import *
from custom_types import DaRnnNet, TrainData, TrainConfig
from utils import numpy_to_tvar
from constants import device



encoder = Encoder(26350,64,10)
encoder.load_state_dict(torch.load("data/encoder.torch"))
encoder.eval()

print("Model's state_dict:")
for param_tensor in encoder.state_dict():
  print(param_tensor, "\t", encoder.state_dict()[param_tensor].size())

relevant_parameters = [i for i,(param_name,param_value) in enumerate(list(encoder.named_parameters())) if 'hh_l0' in param_name]
for i,cur_parameter in enumerate(encoder.parameters()):
    if i in relevant_parameters:
        print("Setting for {0}".format(i))
        cur_parameter.requires_grad=False

save_plots = True
debug = False

raw_data = pd.read_csv(os.path.join("data", "data.csv"), nrows=100 if debug else None)

# For every pixel in area we want a column with its intensity at different times.
logger.info(f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
# print(list(raw_data.columns)) -> first column name is # 1 (TODO: fix)

targ_cols = ('26350',)
data, scaler = preprocess_data(raw_data, targ_cols)

Encoder.forward(data)
exit()

decoder = Decoder(64,64,10)
decoder.load_state_dict(torch.load("data/decoder.torch"))
decoder.eval()

print("Model's state_dict:")
for param_tensor in decoder.state_dict():
    print(param_tensor, "\t", decoder.state_dict()[param_tensor].size())

relevant_parameters2 = [i for i,(param_name,param_value) in enumerate(list(decoder.named_parameters())) if 'hh_l0' in param_name]
for i,cur_parameter in enumerate(decoder.parameters()):
    if i in relevant_parameters2:
        print("Setting for {0}".format(i))
        cur_parameter.requires_grad=False

learning_rate = 0.01
encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=learning_rate)
decoder_optimizer = optim.Adam(
    params=[p for p in decoder.parameters() if p.requires_grad],
    lr=learning_rate)

model1 = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)


save_plots = True
debug = False

raw_data = pd.read_csv(os.path.join("data", "data.csv"), nrows=100 if debug else None)

# For every pixel in area we want a column with its intensity at different times.
logger.info(f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
# print(list(raw_data.columns)) -> first column name is # 1 (TODO: fix)

targ_cols = ('26350',)
data, scaler = preprocess_data(raw_data, targ_cols)

da_rnn_kwargs = {"batch_size": 128, "T": 10}

# config, model = da_rnn(data, n_targs=len(targ_cols), learning_rate=.001, **da_rnn_kwargs)

config = TrainConfig(10, int(data.feats.shape[0] * 0.7), 128, nn.MSELoss())
logger.info(f"Training size: {config.train_size:d}.")


iter_loss, epoch_loss = train(model, data, config, n_epochs=10, save_plots=save_plots)
final_y_pred = predict(model1, data, config.train_size, config.batch_size, config.T)

plt.figure()
plt.semilogy(range(len(iter_loss)), iter_loss)
utils.save_or_show_plot("iter_loss.png", save_plots)

plt.figure()
plt.semilogy(range(len(epoch_loss)), epoch_loss)
utils.save_or_show_plot("epoch_loss.png", save_plots)

plt.figure()
plt.plot(final_y_pred, label='Predicted')
plt.plot(data.targs[config.train_size:], label="True")
plt.legend(loc='upper left')
utils.save_or_show_plot("final_predicted.png", save_plots)

with open(os.path.join("data", "da_rnn_kwargs.json"), "w") as fi:
    json.dump(da_rnn_kwargs, fi, indent=4)

joblib.dump(scaler, os.path.join("data", "scaler.pkl"))
#torch.save(model.encoder.state_dict(), os.path.join("data", "encoder.torch"))
#torch.save(model.decoder.state_dict(), os.path.join("data", "decoder.torch"))
