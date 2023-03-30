import typing
from typing import Tuple
import json
import os

import torch
import joblib
from torch import nn
from torch import optim
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utils
from modules import *
from main import *
from modules import Encoder, Decoder
from custom_types import DaRnnNet, TrainData, TrainConfig
from utils import numpy_to_tvar
from constants import device



encoder_state_dict = torch.load(os.path.join("data", "encoder.torch"))
decoder_state_dict = torch.load(os.path.join("data", "decoder.torch"))

raw_data = pd.read_csv(os.path.join("data", "data2.csv"))

targ_cols = ()
data, scaler = preprocess_data(raw_data, targ_cols)

# enc_kwargs = {"input_size": train_data.feats.shape[1], "hidden_size": encoder_hidden_size, "T": T}

# Instantiate encoder model
encoder = Encoder(data.feats.shape[1], 64, 10)

#  dec_kwargs = {"encoder_hidden_size": encoder_hidden_size, "decoder_hidden_size": decoder_hidden_size, "T": T, "out_feats": n_targs}

# Instantiate decoder model
decoder = Decoder(64,64, 10, 1)

encoder.load_state_dict(encoder_state_dict)
decoder.load_state_dict(decoder_state_dict)


# Process new data with the encoder
with torch.no_grad():
    encoded_data = encoder(data.feats)

# Use the decoder to generate output from the encoded data
with torch.no_grad():
    output = decoder(encoded_data)