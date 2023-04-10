from PIL import Image
import numpy as np
from skimage.io import imsave

import os
import tifffile
import numpy as np
from matplotlib import pyplot as plt

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
    
    imsave('resized_video.tiff', np.array(resized_video))

resize_video("./data/08_baseline_norm.tif", "./data/output.tif", 91, 72)