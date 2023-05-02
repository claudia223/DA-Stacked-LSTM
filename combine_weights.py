import os
import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import tifffile as tiff


#The sub_folders containing our attention weights
sub_folders= ["results_0", "results_1","results_2","results_3","results_4","results_5", "results_6" ]
#sub_folders= ["results_0","results_1"]


# Define a function to create a heatmap plot with the original video as background
def plot_heatmap_with_video(frame, data, video):
    # plot the original video frame as the background
    plt.imshow(video[frame], cmap='gray')
    
    #height, width = video[frame].shape
    # plot the heatmap as a transparent overlay on top of the original video frame
    # in reshape the values are (height, width)
    plt.imshow(data, alpha=0.5, cmap='coolwarm')

    # add a colorbar to the plot
    plt.colorbar()

## For making the heatmap
def update(frame):
    plt.clf()  # Clear the previous plot
  
    # we have tensor of shape torch.Size([398]), we need 20*20
    new_frame = weight_tensor[frame,:,:]
    #new_frame = new_frame.detach().numpy()


    # Plot the heatmap and video for the current frame
    plot_heatmap_with_video(frame, np.absolute(new_frame), video)
    
    plt.title('Frame {}'.format(frame))
    plt.tight_layout()



"""

If the sweep has been split width-wise, after running the model and obtaining the attention weights, 
(stored under weight_tensor), we can concatenate these weights using the function
this function. The weights are resized back to their original shape and the target pixel is added
back to form the heatmap (our model doesn't give an attention weight to the target pixel which means
that we need to initiate it to 0 ourselves).

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



"""
def concat_weight_tensors(root_folder, frameNo=8700, height=72, width=13):
    reshaped_frames = []
    for subdir in sub_folders:
        weight_file = os.path.join(root_folder, subdir, "weight_tensor_rs.pt")
        if os.path.exists(weight_file):
            weight_tensor = torch.load(weight_file)
            mean_tensor = weight_tensor[:,1,:] 
            new_frame = mean_tensor.detach().numpy()

            if subdir == "results_2":
                # create a new column with zeros
                new_column = np.zeros((1,frameNo))
                # insert the new column at the 220th position (index 219)
                #print("Before: {}".format(new_frame.shape))
                new_frame = np.insert(new_frame, 219, new_column, axis=1)
                #print(new_frame.shape)

            reshaped_frame= new_frame.reshape(frameNo,height, width)


            #print(reshaped_frame.shape)
            reshaped_frames.append(reshaped_frame)


    # Concatenate all weight tensors into one big tensor along the first dimension (frames)
    concatenated_weights = np.concatenate(reshaped_frames, axis=2)
    
    

    print(concatenated_weights.shape)

    
    return concatenated_weights


weight_tensor= concat_weight_tensors("./results/")
video= tiff.imread('./time_cropped.tiff')

# Create the animation
fig = plt.figure(figsize=(6, 6))

#Frames need to be numbver of frames we want in our animation
ani = animation.FuncAnimation(fig, update, frames=8700, interval=200)
ani.save('heatmap_final.mp4', writer='ffmpeg', bitrate=2000)
