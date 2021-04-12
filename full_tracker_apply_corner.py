#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 18:01:57 2021

@author: larsmaurath
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

import cv2

from narya.narya.tracker.full_tracker import FootballTracker
from narya.narya.utils.vizualization import visualize

#%%

template = cv2.imread('narya/world_cup_template.png')
template = cv2.resize(template, (512,512))/255.


#%%
# import matplotlib.pyplot as plt

# image = cv2.imread("atm_che_23022021_62_07_2.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# image = image[612:712, 337:387]

# pixels = np.float32(image.reshape(-1, 3))

# n_colors = 5
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
# flags = cv2.KMEANS_RANDOM_CENTERS

# _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags) 
# _, counts = np.unique(labels, return_counts=True)

#%%
# dominant = palette[np.argmax(counts)]

# avg_patch = np.ones(shape=image.shape, dtype=np.uint8)*np.uint8(dominant)

# cv2.rectangle(image,(612, 712),(337, 387),tuple(dominant.tolist()),3)


# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,6))
# ax0.imshow(image)
#%%
    
image = cv2.imread("atm_che_23022021_62_07_2.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img_list = [image]
        
# image_0,image_25,image_50 = img_list[0],img_list[25],img_list[50]
# print("Image shape: {}".format(image_0.shape))
# visualize(image_0=image_0,
#             image_25=image_25,
#             image_50=image_50)


#%% 

tracker = FootballTracker(pretrained=True, 
                          frame_rate=23,
                          track_buffer = 60,
                          ctx=None)

#%%

trajectories = tracker(img_list,
                       split_size = 512, 
                       save_tracking_folder = 'narya_output/', 
                       template = template, 
                       skip_homo = [])


#%%
# path = os.path.expanduser('test_outputs/')
# files = os.listdir(path)
# files = [x for x in files if x != '.DS_Store']
# files.sort()

# import imageio
# import progressbar
# with imageio.get_writer('test_outputs/movie.mp4', mode='I',fps=20) as writer:
#     for i in progressbar.progressbar(range(51)):
#         image = imageio.imread(os.path.join(path, files[i]))
#         writer.append_data(image)