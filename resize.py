from PIL import Image
import numpy as np
from skimage.io import imsave

import os
import tifffile
import numpy as np
from matplotlib import pyplot as plt


def convert_csv(video_np):
    n_frames, height, width = video_np.shape
    video_np = np.reshape(video_np, (n_frames, height*width))
    print(video_np.shape)


    head = [str(i) for i in range(1,(height*width))]

    np.savetxt("resized_data.csv", video_np, delimiter=",", header=','.join(str(elem) for elem in head))


def reader(sample_dir):
    sample_np = tifffile.memmap(sample_dir, mode='r')
    timepoint = 0
    max_timepoints = sample_np.shape[0]
    
    return sample_np

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

if not os.path.exists("resized_video.tiff") or not os.path.exists("resized.npy"):
    new_video= resize_video("./data/08_baseline_norm.tif", "resized_video.tiff", 91, 72)
else:
    new_video= np.load("resized.npy")

convert_csv(new_video)

