# DA-Stacked-LSTMs for CSD detection
This repository contains code for using the implementation of the Dual-Attention Based Stacked LSTMs (DA-Stacked LSTMs) model to detect Cortical Spreading Depression (CSD) events from raw calcium imaging data. The code is adapted from the following implementation of the paper https://github.com/Seanny123/da-rnn. The main objective is to identify the brain regions that are most significant for the a target pixel.

We have therefore preserved the attention weights and generated a heatmap based on them. It is important to mention that we shuffled the columns, i.e. the pixels for each frame, at the beginning of each epoch. This was done to ensure that we obtained the most useful pixels instead of just the nearest ones. By overlaying this heatmap on the original sample, we can examine the correlation between the brain's electrical activity and the significance that the model has attributed to it. 

![Gifs can be viewed from the results folder if they do not work here](heatmap_final_gif.gif)

![Gifs can be viewed from the results folder if they do not work here](heatmap_new_gif.gif)

## Installation & Set-up 
1. Clone the repository:  
```
https://github.com/claudia223/DA-Stacked-LSTM
```
2. Navigate to the repository
##


## Dataset

The dataset to be run by the model will be stored in the ./data folder. We expect a TIFF file to be in the folder, but the file format can be changed in the file "resize.py". Here "insert_video.tiff" can be replaced by any type of video file. 
```sh
python3  resize.py

```


```python
if not os.path.exists("insert_video.tiff") or not os.path.exists("resized.npy"):
    new_video= resize_video("./data/insert_video.tif", "resized_video.tiff", 91, 72)
else:
    new_video= np.load("resized.npy")

```

The inputted video sweep will need to be converted into a CSV which is done by the convert_csv function in resize.py. You can change the output csv's name to what you prefer, in the savetext command. Each frame is resized from 2D into 1D, and stored as a row in the CSV. The number of rows in the CSV correspond to the number of timeframes in the video.

```python
def convert_csv(video_np):
    n_frames, height, width = video_np.shape
    #half_n_frames = (n_frames // 10)
    video_np = np.reshape(video_np, (n_frames, height*width))
    #video_np_second_half = video_np[half_n_frames:, :]
    #print(video_np_second_half.shape)


    head = [str(i) for i in range(1,(height*width))]

    np.savetxt("resized_data.csv", video_np, delimiter=",", header=','.join(str(elem) for elem in head))

```

To obtain a cropped version of the dataset (such as the 20x20 pixel crop mentioned in the report), the resize_video function can be used. This function accepts the following parameters:

- video_dir: The directory of the video that will be used on the model.
- output_dir: The output directory of the resized video
- new_width: The new width of the video
- new_height: The new height of the video

The function saves a copy of the resized video as well as the numpy (which is of the same shape as the CSV mentioned previously). If we want the csv, we can just run the aforementioned convert_csv function.

If there is need to crop the video in time, the crop_tiff_time is also available. The crop can be selected by modifying the starting index.

If we want to split a sample, width wise, the function crop_numpy_width(numpy_array, output_dir, target_col, number_splits) can be used. Given a numpy of the video, it can split the video "number_splits" times. For instance, if number_splits is 7, then the output will be stored as 7 csv(s) and 7 tiff(s).


## Running the model

To run the model, we can use:

```
python3 main.py

```

The most important function is da_rnn:

```python
def da_rnn(train_data: TrainData, n_targs: int, encoder_hidden_size=64, decoder_hidden_size=64,
           T=10, learning_rate=0.01, batch_size=128):

```

 Most of the important parameters for our model can be modified here:

- train_data: This parameter is an instance of the TrainData class and contains the training data that will be used to train the model.

- n_targs: This parameter is an integer that specifies the number of target variables in the training data. In other words, it specifies the number of variables that the model is trying to predict.

- encoder_hidden_size: This parameter is an integer that specifies the number of hidden units in the encoder LSTM layer of the model.

- decoder_hidden_size: This parameter is an integer that specifies the number of hidden units in the decoder LSTM layer of the model.

- T: This parameter is an integer that specifies the window sized used for prediction.

- learning_rate: This parameter is a float that specifies the learning rate to be used during training.

- batch_size: This parameter is an integer that specifies the size of the mini-batches to be used during training.


When main.py is compiled, the function run_all() is called. 

```python

def run_all(data="splits/crop_0.csv", targ_cols= ('937',) ,  weight_tensor_dir= "weight_tensor.pt", iter_loss_dir="iter_loss.npy" , \
             epoch_loss_dir= "epoch_loss.npy", final_y_pred_dir="final_y_pred.npy" ):
    

```

Here, it is possible to specify where the sweep's directory (has to be a csv), the target column (this refers to the pixel position in a 1D array (i.e. if the pixel was initially in coordinate (12,20), then the target pixel would be 240 in 1D)). The rest of the parameters are used in case the weights have already been calculated and we don't want to run the entire model again.


To obtain the plots for our prediction, the create_plots function can be run.
This will create plots for the iteration loss, which is the MSE loss over each *training* iteration, the epoch loss, which is the MSE loss over each epoch and the final_prediction which shows the predicted pixel intensity over the true pixel value.


To obtain the heatmap of the attention weights to see which regions are the most impactful, we can use:

```python
fig = plt.figure(figsize=(6, 6))
#Frames need to be numbvr of frames we want in our animation
ani = animation.FuncAnimation(fig, update, frames=8700, interval=200)
ani.save('heatmap.mp4', writer='ffmpeg', bitrate=1000)
```

## Combining weights

If the sweep has been split width-wise, after running the model and obtaining the attention weights, (stored under weight_tensor), we can concatenate these weights using the function concat_weight_tensors in combine_weights.py.

The expected subfolder structure containing the weights are the following:

```
results/
├── results_1/
│   ├── plots/
│   │   ├── epoch_loss.png/
│   │   ├── final_predicted.png/
│   │   └── iter_loss.png/
│   └── weight_tensor_rs.pt
├── results_2/
│   ├── plots/
│   │   ├── epoch_loss.png/
│   │   ├── final_predicted.png/
│   │   └── iter_loss.png/
│   └── weight_tensor_rs.pt
├── results_3/
├── results_4/
└── ...

```

A heatmap of these combined weights can be produced by running

```
python3 combine_weights.py

```

