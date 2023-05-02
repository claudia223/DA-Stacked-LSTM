import typing
from typing import Tuple
import json
import os

import torch
import joblib
from torch import nn
from torch import optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import numpy as np
import tifffile as tiff

import utils
from modules import Encoder, Decoder
from custom_types import DaRnnNet, TrainData, TrainConfig
from utils import numpy_to_tvar
from constants import device

logger = utils.setup_log()
logger.info(f"Using computation device: {device}")

"""
Based on the original DA-RNN encoder and decoder network: 

ttps://github.com/Seanny123/da-rnn

"""
def preprocess_data(dat, col_names) -> Tuple[TrainData, StandardScaler]:
    print(col_names)
    scale = StandardScaler().fit(dat)
    proc_dat = scale.transform(dat)

    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)

    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False

    feats = proc_dat[:, mask]
    targs = proc_dat[:, ~mask]

    return TrainData(feats, targs), scale


def da_rnn(train_data: TrainData, n_targs: int, encoder_hidden_size=64, decoder_hidden_size=64,
           T=10, learning_rate=0.01, batch_size=128):

    train_cfg = TrainConfig(T, int(train_data.feats.shape[0] * 0.7), batch_size, nn.MSELoss())
    logger.info(f"Training size: {train_cfg.train_size:d}.")

    enc_kwargs = {"input_size": train_data.feats.shape[1], "hidden_size": encoder_hidden_size, "T": T}
    encoder = Encoder(**enc_kwargs).to(device)
    with open(os.path.join("data", "enc_kwargs.json"), "w") as fi:
        json.dump(enc_kwargs, fi, indent=4)

    dec_kwargs = {"encoder_hidden_size": encoder_hidden_size,
                  "decoder_hidden_size": decoder_hidden_size, "T": T, "out_feats": n_targs}
    decoder = Decoder(**dec_kwargs).to(device)
    with open(os.path.join("data", "dec_kwargs.json"), "w") as fi:
        json.dump(dec_kwargs, fi, indent=4)

    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=learning_rate)
    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad],
        lr=learning_rate)
    da_rnn_net = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)

    return train_cfg, da_rnn_net


def train(net: DaRnnNet, train_data: TrainData, t_cfg: TrainConfig, n_epochs=10, save_plots=False):
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    epoch_losses = np.zeros(n_epochs)
    logger.info(f"Iterations per epoch: {t_cfg.traToin_size * 1. / t_cfg.batch_size:3.3f} ~ {iter_per_epoch:d}.")

    n_iter = 0

    for e_i in range(n_epochs):
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)

        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):
            batch_idx = perm_idx[t_i:(t_i + t_cfg.batch_size)]
            feats, y_history, y_target = prep_train_data(batch_idx, t_cfg, train_data)

            loss = train_iteration(net, t_cfg.loss_func, feats, y_history, y_target)
            iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = loss
            # if (j / t_cfg.batch_size) % 50 == 0:
            #    self.logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / t_cfg.batch_size, loss)
            n_iter += 1

            adjust_learning_rate(net, n_iter)

        epoch_losses[e_i] = np.mean(iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])


        #if e_i % 5 == 0 
        if (e_i % 5 == 0):
            
            y_test_pred,_ = predict(net, train_data,
                                  t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                  on_train=False)
        
            y_train_pred,_ = predict(net, train_data,
                                    t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                    on_train=True)
                
            # TODO: make this MSE and make it work for multiple inputs
            val_loss = y_test_pred - train_data.targs[t_cfg.train_size:]
            logger.info(f"Epoch {e_i:d}, train loss: {epoch_losses[e_i]:3.3f}, val loss: {np.mean(np.abs(val_loss))}.")


            plt.figure()
            plt.plot(range(1, 1 + len(train_data.targs)), train_data.targs,
                     label="True") #color= "blue"
            plt.plot(range(t_cfg.T, len(y_train_pred) + t_cfg.T), y_train_pred,
                     label='Predicted - Train') # color= "orange"
            plt.plot(range(t_cfg.T + len(y_train_pred), len(train_data.targs) + 1), y_test_pred,
                     label='Predicted - Test') # color= 'green'
            plt.legend(loc='upper left')
            utils.save_or_show_plot(f"pred_{e_i}.png", save_plots)

    return iter_losses, epoch_losses


def prep_train_data(batch_idx: np.ndarray, t_cfg: TrainConfig, train_data: TrainData):
    feats = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.feats.shape[1]))
    y_history = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.targs.shape[1]))
    y_target = train_data.targs[batch_idx + t_cfg.T]

    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx, b_idx + t_cfg.T - 1)
        feats[b_i, :, :] = train_data.feats[b_slc, :]
        y_history[b_i, :] = train_data.targs[b_slc]

    return feats, y_history, y_target


def adjust_learning_rate(net: DaRnnNet, n_iter: int):
    # TODO: Where did this Learning Rate adjustment schedule come from?
    # Should be modified to use Cosine Annealing with warm restarts https://www.jeremyjordan.me/nn-learning-rate/
    if n_iter % 10000 == 0 and n_iter > 0:
        for enc_params, dec_params in zip(net.enc_opt.param_groups, net.dec_opt.param_groups):
            enc_params['lr'] = enc_params['lr'] * 0.9
            dec_params['lr'] = dec_params['lr'] * 0.9


def train_iteration(t_net: DaRnnNet, loss_func: typing.Callable, X, y_history, y_target):
    t_net.enc_opt.zero_grad()
    t_net.dec_opt.zero_grad()

    _, input_encoded = t_net.encoder(numpy_to_tvar(X))
    y_pred = t_net.decoder(input_encoded, numpy_to_tvar(y_history))

    y_true = numpy_to_tvar(y_target)
    loss = loss_func(y_pred, y_true)
    loss.backward()

    t_net.enc_opt.step()
    t_net.dec_opt.step()

    return loss.item()


def predict(t_net: DaRnnNet, t_dat: TrainData, train_size: int, batch_size: int, T: int, on_train=False):
    out_size = t_dat.targs.shape[1]
    input_weighted = []
    count = 0
    if on_train:
        y_pred = np.zeros((train_size - T + 1, out_size))
    else:
        print(batch_size)
        y_pred = np.zeros((t_dat.feats.shape[0] - train_size, out_size))

        print(len(y_pred))

    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)
        X = np.zeros((b_len, T - 1, t_dat.feats.shape[1]))
        y_history = np.zeros((b_len, T - 1, t_dat.targs.shape[1]))

        for b_i, b_idx in enumerate(batch_idx):
            if on_train:
                idx = range(b_idx, b_idx + T - 1)
            else:
                idx = range(b_idx + train_size - T, b_idx + train_size - 1)

            X[b_i, :, :] = t_dat.feats[idx, :]
            y_history[b_i, :] = t_dat.targs[idx]

        y_history = numpy_to_tvar(y_history)
        
        input_weight, input_encoded = t_net.encoder(numpy_to_tvar(X))
        if not on_train:
            input_weighted.append(input_weight)

        y_pred[y_slc] = t_net.decoder(input_encoded, y_history).cpu().data.numpy()

    return y_pred, input_weighted


save_plots = True
debug = False

"""

 Most of the important parameters for our model can be modified here:

    - train_data: This parameter is an instance of the TrainData class and
    contains the training data that will be used to train the model.

    - n_targs: This parameter is an integer that specifies the number of target
    variables in the training data. In other words, it specifies the number of variables 
    that the model is trying to predict.

    - encoder_hidden_size: This parameter is an integer that specifies the number 
    of hidden units in the encoder LSTM layer of the model.

    - decoder_hidden_size: This parameter is an integer that specifies the number of hidden units
    in the decoder LSTM layer of the model.

    - T: This parameter is an integer that specifies the window sized used for prediction.

    - learning_rate: This parameter is a float that specifies the learning rate to be used 
    during training.

    - batch_size: This parameter is an integer that specifies the size of the mini-batches
    to be used during training.




"""


def run_all(data="splits/crop_0.csv", targ_cols= ('937',) ,  weight_tensor_dir= "weight_tensor.pt", iter_loss_dir="iter_loss.npy" , \
             epoch_loss_dir= "epoch_loss.npy", final_y_pred_dir="final_y_pred.npy" ):
    
    raw_data = pd.read_csv(data)

    # For every pixel in area we want a column with its intensity at different times.
    logger.info(f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")

    # Thes are the columns we want to use as labels
    
    data, scaler = preprocess_data(raw_data, targ_cols)

    # We can change batch_size and T
    da_rnn_kwargs = {"batch_size": 20, "T": 10}
    config, model = da_rnn(data, n_targs=len(targ_cols), learning_rate=.01, **da_rnn_kwargs)
    
    if os.path.exists(weight_tensor_dir):
        print("NO")
        weight_tensor = torch.load(weight_tensor_dir)
        iter_loss = np.load(iter_loss_dir)
        epoch_loss = np.load(epoch_loss_dir)
        final_y_pred = np.load(final_y_pred_dir)

    else:
        iter_loss, epoch_loss = train(model, data, config, n_epochs=5, save_plots=save_plots)
    
    
    final_y_pred, input_weighted = predict(model, data, config.train_size, config.batch_size, config.T)


    weight_tensor = torch.cat(input_weighted, dim=0)
    
    torch.save(weight_tensor, weight_tensor_dir)
    np.save(iter_loss_dir, iter_loss)
    np.save(epoch_loss_dir, epoch_loss)
    np.save(final_y_pred_dir,final_y_pred)

    print(weight_tensor.shape)
    create_plots(iter_loss, epoch_loss, final_y_pred, data, config)

    

 
    
    


def create_plots(iter_loss, epoch_loss, final_y_pred, data, config):
    plt.figure()
    plt.semilogy(range(len(iter_loss)), iter_loss)
    utils.save_or_show_plot("iter_loss.png", save_plots)

    plt.figure()
    plt.semilogy(range(len(epoch_loss)), epoch_loss)
    utils.save_or_show_plot("epoch_loss.png", save_plots)

    plt.figure()
    plt.plot(data.targs[config.train_size:], label="True") #,color= "blue"
    plt.plot(final_y_pred, label='Predicted') # color= "orange"
    plt.legend(loc='upper left')
    utils.save_or_show_plot("final_predicted.png", save_plots)


video = tiff.imread('./data/test.tif')

# Load your video here and assign it to a variable "video"
# video = load_video('your_video.mp4') # replace 'your_video.mp4' with the path to your video file

# Define a function to create a heatmap plot with the original video as background
def plot_heatmap_with_video(frame, data, video):
    # plot the original video frame as the background
    plt.imshow(video[frame], cmap='gray')
    
    height, width = video[frame].shape
    # plot the heatmap as a transparent overlay on top of the original video frame
    # in reshape the values are (height, width)
    plt.imshow(data.reshape((height, width)), alpha=0.5, cmap='coolwarm')

    # add a colorbar to the plot
    plt.colorbar()

# Define a function to update the heatmap and input for each frame of the animation
# Weight tensor as global variable TODO: Change this.
def update(frame):
    plt.clf()  # Clear the previous plot
  
    # we have tensor of shape torch.Size([398]), we need 20*20
    new_frame = weight_tensor[frame,1,:]
    new_frame = new_frame.detach().numpy()
    # we now add two new values at at 199?? and 200 (yes)
    
    #new_frame = np.insert(new_frame,1892,0)
    new_frame = np.insert(new_frame,190,0)
    # Plot the heatmap and video for the current frame
    plot_heatmap_with_video(frame, np.absolute(new_frame), video)
    
    plt.title('Frame {}'.format(frame))
    plt.tight_layout()



# shape weights= (8700, 936)

run_all("./splits/crop_4.csv",('936',),"./results_4/weight_tensor_rs.pt", "iter_loss_rs.npy" , \
             "epoch_loss_rs.npy", "final_y_pred_rs.npy" )


# Create the animation
fig = plt.figure(figsize=(6, 6))
# Frames need to be numbver of frames we want in our animation
#ani = animation.FuncAnimation(fig, update, frames=8700, interval=200)
#ani.save('heatmap.mp4', writer='ffmpeg', bitrate=1000)