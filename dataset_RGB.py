import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif','raw',"bin"])
def read_bin(train_path, test=False):
    if not test:
        content = open(train_path, 'rb').read()
        samples_ref = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
        # print("train_path:", train_path)
        label = train_path.replace("_input", "_gt")
        content = open(label, 'rb').read()
        samples_gt = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
        inp_filenames = [samples_ref[i, ...] for i in range(samples_ref.shape[0])]
        tar_filenames = [samples_gt[i, ...] for i in range(samples_gt.shape[0])]
        return inp_filenames, tar_filenames
    else:
        content = open(train_path, 'rb').read()
        samples_ref = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
        # print("train_path:", train_path)
        inp_filenames = [samples_ref[i, ...] for i in range(samples_ref.shape[0])]
        return inp_filenames, samples_ref,content
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, maskdir, img_options=None):
        super(DataLoaderTrain, self).__init__()
        self.inp_filenames = rgb_dir
        self.tar_filenames = maskdir

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

        self.patchsx = self.img_options['patch_size']
        self.patchsy = self.img_options['patch_size']
    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        # 将我们读取出来的raw文件内容放入到我们创建的文件当中
        inp_img_32 = np.zeros((self.patchsx, self.patchsy), dtype='float32')
        inp_img_32[:, :] = np.float32(inp_path[:, :]) * np.float32(1 / 65536)

        tar_img_32 = np.zeros((self.patchsx, self.patchsy), dtype='float32')
        tar_img_32[:, :] = np.float32(tar_path[:, :]) * np.float32(1 / 65536)
        # inp_img = Image.open(inp_path)
        # tar_img = Image.open(tar_path)

        inp_img = inp_img_32
        tar_img = tar_img_32
        inp_img = inp_img[None, ...]
        tar_img = tar_img[None, ...]

        inp_img = torch.from_numpy(inp_img)
        tar_img = torch.from_numpy(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]
        # print(tar_img.shape)
        rr     = random.randint(0, hh-ps)
        cc     = random.randint(0, ww-ps)
        aug    = random.randint(0, 8)

        # Crop patch

        inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]

        # Data Augmentations
        if aug==1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug==2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug==3:
            inp_img = torch.rot90(inp_img,dims=(1,2))
            tar_img = torch.rot90(tar_img,dims=(1,2))
        elif aug==4:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
        elif aug==5:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
        elif aug==6:
            inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
        elif aug==7:
            inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))
        if inp_img.shape[0] != 1 or inp_img.shape[1] != 256 or inp_img.shape[2] != 256:
            print(inp_img.shape)
            print("aug:",aug)
        # filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img

class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, maskdir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()


        self.inp_filenames = rgb_dir
        self.tar_filenames = maskdir

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']
        self.patchsx = self.img_options['patch_size']
        self.patchsy = self.img_options['patch_size']
    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        # 将我们读取出来的raw文件内容放入到我们创建的文件当中
        inp_img_32 = np.zeros((self.patchsx, self.patchsy), dtype='float32')
        inp_img_32[:, :] = np.float32(inp_path[:, :]) * np.float32(1 / 65536)

        tar_img_32 = np.zeros((self.patchsx, self.patchsy), dtype='float32')
        tar_img_32[:, :] = np.float32(tar_path[:, :]) * np.float32(1 / 65536)


        inp_img = inp_img_32[None, ...]
        tar_img = tar_img_32[None, ...]
        inp_img = torch.from_numpy(inp_img)
        tar_img = torch.from_numpy(tar_img)
        if inp_img.shape[0] != 1 or inp_img.shape[1] != 256 or inp_img.shape[2] != 256:
            print(inp_img.shape)
        # filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        self.inp_filenames = inp_dir

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

        self.patchsx = self.img_options['patch_size']
        self.patchsy = self.img_options['patch_size']
    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        inp_img_32 = np.zeros((self.patchsx, self.patchsy), dtype='float32')
        inp_img_32[:, :] = np.float32(path_inp[:, :]) * np.float32(1 / 65536)
        inp_img = inp_img_32[None, ...]

        inp_img = torch.from_numpy(inp_img)
        return inp_img
