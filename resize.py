from PIL import Image
import numpy as np
from skimage.io import imsave

import os
import tifffile
import numpy as np
from matplotlib import pyplot as plt
import imageio
from PIL import Image
import numpy as np
import pandas as pd
import imageio
import csv


"""

The inputted video sweep will need to be converted into a CSV which is done using this function.
 You can change the output csv's name to what you prefer, in the savetext command. 
 Each frame is resized from 2D into 1D, and stored as a row in the CSV. 
 The number of rows in the CSV correspond to the number of timeframes in the video.

"""
def convert_csv(video_np):
    n_frames, height, width = video_np.shape
    video_np = np.reshape(video_np, (n_frames, height*width))



    head = [str(i) for i in range(1,(height*width))]

    np.savetxt("resized_data_7.csv", video_np, delimiter=",", header=','.join(str(elem) for elem in head))



"""
To obtain a cropped version of the dataset (such as the 20x20 pixel crop mentioned in the report), the resize_video function can be used. This function accepts the following parameters:

- video_dir: The directory of the video that will be used on the model.
- output_dir: The output directory of the resized video
- new_width: The new width of the video
- new_height: The new height of the video

The function saves a copy of the resized video as well as the numpy (which is of the same shape as the CSV mentioned previously). 
If we want the csv, we can just run the aforementioned convert_csv function.

Reduced using bilinear interpolation.

"""

def resize_video(video_dir, output_dir, new_width, new_height):
    video = reader(video_dir)
    #print(video.shape)
    resized_video = []
    for frame in video:
        img = Image.fromarray(frame)
        resized_img = img.resize((new_width, new_height), resample=Image.BILINEAR)
        resized_video.append(np.array(resized_img))
    
    np_video= np.array(resized_video)
    imsave(output_dir, np_video)
    np.save("resized.npy", np_video)
    
    return np_video


"""
If we want to split a sample, width wise, the functio can be used. 
Given a numpy of the video, it can split the video "number_splits" times.
For instance, if number_splits is 7, then the output will be stored as 7 csv(s) and 7 tiff(s).

"""


def crop_numpy_width(numpy_array, output_dir, target_col, number_splits):
    # Calculate the width of each cropped array
    crop_width = numpy_array.shape[2] // 7
    print(crop_width)
    

    # Iterate over the 7 cropped arrays
    for i in range(number_splits):
        # Calculate the left and right indices of the crop
        left = i * crop_width
        right = (i + 1) * crop_width

        # Crop the array
        crop = numpy_array[:, :, left:right]

        np_video= np.array(crop)
        vid_output= f"{output_dir}/video_{i}.tiff"
        imsave(vid_output, np_video)
    


        # Reshape the crop into a 2D array with shape (29000, crop_width * 72)
        crop_2d = np.reshape(crop, (crop.shape[0], -1))
        
        if i != 2:
            target_2d = target_col.reshape(-1, 1)
            #print(target_2d.shape)
            crop_2d = np.hstack((crop_2d, target_2d))
        
        print(crop_2d.shape)
        header = [i for i in range(crop_2d.shape[1])]
        # Convert the 2D array to a DataFrame
        df = pd.DataFrame(crop_2d, columns=header)

        

        # Save the DataFrame as a CSV file
        csv_path = f"{output_dir}/crop_{i}.csv"
        df.to_csv(csv_path, index=False)


"""
If there is need to crop the video in time, this function is also available. 
The crop can be selected by modifying the starting index.

"""

def crop_tiff_time(video_path, output_path):
    # Read the video frames using imageio
    video = imageio.get_reader(video_path)

    # Get the number of frames in the video
    num_frames = len(video)

    # Calculate the starting index for the crop
    start_idx = num_frames - 8700

    # Iterate over the frames in the video
    cropped_frames = []
    for i, frame in enumerate(video):
        # Only keep the frames after the starting index
        if i >= start_idx:
            cropped_frames.append(frame)

    # Save the cropped frames as a new tiff video
    imsave(output_path, np.array(cropped_frames))
    #imageio.mimsave(output_path, cropped_frames)






if not os.path.exists("resized_video.tiff") or not os.path.exists("resized.npy"):
    new_video= resize_video("./data/08_baseline_norm.tif", "resized_video.tiff", 91, 72)
else:
    new_video= np.load("resized.npy")


target_col= new_video[:, 20, 25] 
crop_numpy_width(new_video, "./splits", target_col, 7)

#To convert a video into a csv:
#convert_csv(new_video)

#To crop in time:
#crop_tiff_time('./splits/video_0.tiff', './splits/time_video_0.tiff')



