import numpy as np
# import cv2 #https://www.jianshu.com/p/f2e88197e81d
import random

from skimage.io import imread
from skimage import color

import torch
import torch.utils.data
from torchvision import datasets, models, transforms
import os
import random
class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        data_len = len(img_paths)
        begin = data_len // 6
        self.img_path = []
        self.mask_path = []
        for i in range(0, 6):
            self.img_path.append(img_paths[i*begin:begin*i+begin])
            self.mask_path.append(mask_paths[i*begin:begin*i+begin])
        num = random.randint(0, 5)
        self.img_paths =  self.img_path[num]
        # self.mask_paths = list(map(lambda x: x.replace('volume', 'segmentation').replace('image','mask'), self.img_paths))
        self.mask_paths = self.mask_path[num]
        self.aug = aug

        # print(self.img_paths,self.mask_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        idex = img_path.split("/")[-1].split(".")[0].split("-")[-1]
        mask_path = self.mask_paths[idx]
        last_mask_name = mask_path.split('/')[-1]
        first_mask_name = mask_path.split('/seg')[0]
        mask_idex = mask_path.split("/")[-1].split(".")[0].split("-")[-1]
        new_last_mask_name = last_mask_name.replace(mask_idex, idex)
        mask_path = os.path.join(first_mask_name, new_last_mask_name)
        npimage = np.load(img_path)
        npmask = np.load(mask_path)
        # print("img:{},mask:{}".format(npimage.shape, npmask.shape))
        npimage = npimage[:, :, :, np.newaxis]
        npimage = npimage.transpose((3, 0, 1, 2))
        T = npimage.shape[1]
        H = npimage.shape[2]
        W = npimage.shape[3]
        # print("T:{},H:{},W:{}".format(T,H,W))
        image = np.zeros((1, 64,160,160))
        image[:, 0:T, 0:H, 0:W] = npimage

        liver_label = npmask.copy()
        liver_label[npmask == 2] = 1
        liver_label[npmask == 1] = 1

        tumor_label = npmask.copy()
        tumor_label[npmask == 1] = 0
        tumor_label[npmask == 2] = 1

        nplabel = np.zeros((64,160,160,2))

        nplabel[0:T, 0:H, 0:W, 0] = liver_label
        nplabel[0:T, 0:H, 0:W, 1] = tumor_label


        nplabel = nplabel.transpose((3, 0, 1, 2))
        nplabel = nplabel.astype("float32")
        npimage = image.astype("float32")
        #print(npimage.shape)

        return npimage,nplabel


       