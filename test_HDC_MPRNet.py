"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
from config import Config
opt = Config('training.yml')
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

# from MPRNet import MPRNet
from net.deNoise.HDC_MPRnet import MPRNet
from skimage import img_as_ubyte
from dataset_RGB import DataLoaderTest, read_bin
import h5py
import net
# from Denoising.net.original import Predictor
import scipy.io as sio
from net.layers import get_norm_layer
from pdb import set_trace as stx
from data_RGB import get_test_data
from torch.utils.data import DataLoader
import imageio
# from Denoising.net.DeNoise_HDC import HDC_Net
parser = argparse.ArgumentParser(description='Image Denoising using MPRNet')

parser.add_argument('--input_dir', default='./Datasets/SIDD/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/SIDD/', type=str, help='Directory for results')
parser.add_argument('--resume', default=r'D:\研一下课程\计算机视觉\MPRNet\Denoising\checkpoints\Denoising\models\HDC_MPRNet\model_best.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', default=True, type=bool, help='Save denoised images in result directory')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


# if args.save_images:
#     result_dir_img = os.path.join(args.result_dir, 'png')
#     utils.mkdir(result_dir_img)

model = MPRNet()
# model = HDC_Net()
# model = Predictor()
# model = getattr(net, "Unet")
# model = model(inplanes=1, num_classes=1, width=48, norm_layer=get_norm_layer("group"), deep_supervision=False)

# model = torch.nn.DataParallel(model).cuda()
model = model.cuda()
print(args.resume)
assert os.path.isfile(args.resume),"no checkpoint found at {}".format(args.resume)
print("=> loading checkpoint '{}'".format(args.resume))
checkpoint = torch.load(args.resume)
args.start_iter = checkpoint['epoch']
model.load_state_dict(checkpoint['state_dict'])
msg = ("=> loaded checkpoint '{}' (iter {})".format(args.resume, checkpoint['epoch']))

# utils.load_checkpoint(model_restoration,args.weights)
# # print("===>Testing using weights: ",args.weights)
# model_restoration.cuda()
# model_restoration = nn.DataParallel(model_restoration)
model.eval()

# Process data
test = opt.TRAINING.VAL_DIR
# filepath = os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat')
img_paths, orig_image, content = read_bin(test,test=True)
Z, H, W = orig_image.shape
restored = np.frombuffer(content, dtype = 'uint16').reshape(-1, 256, 256)

paths = r"D:\研一下课程\计算机视觉\MPRNet\Denoising\submit\HDC_MPRNet"
file_paths = r"D:\研一下课程\计算机视觉\MPRNet\Denoising\submit\HDC_MPRNet\result.bin"

if not os.path.exists(paths):
    os.mkdir(paths)
    fout = open(file_paths, 'wb')
    print(fout)
else:
    fout = open(file_paths, 'wb')
# utils.mkdir(fout)
# train_dataset = get_test_data(img_paths, {'patch_size':opt.TRAINING.TRAIN_PS})

batchsz = 16
patchsz = 256
for i, item in enumerate(img_paths):
    batch_inp_np = np.zeros((patchsz, patchsz), dtype = 'float32')
    batch_inp_np[:, :] = np.float32(item[:, :]) * np.float32(1 / 65536)
    patch_tensor = torch.FloatTensor(batch_inp_np).cuda()
    patch_tensor = patch_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
    # print("1",patch_tensor.shape)
    restored_patch = model(patch_tensor)
    restored_patch = restored_patch[0].cpu().detach().numpy()
    # restored_patch = restored_patch.cpu().detach().numpy()
    # print(restored_patch.shape)
    restored_patch = (restored_patch[:, 0, :, :]*65536).clip(0, 65536).astype("uint16")

    # restored_patch = np.squeeze(restored_patch)
    print("i:{},shape:{}".format(i, restored_patch.shape))
    # restored[i, :, :] = restored_patch
    fout.write(restored_patch.tobytes())
fout.close()

