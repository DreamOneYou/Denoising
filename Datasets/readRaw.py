# coding:utf-8
import numpy as np
import torch

# 首先确定原图片的基本信息：数据格式，行数列数，通道数
rows=886#图像的行数
cols=492#图像的列数
patchsx = rows
patchsy = cols
batchsz = 16
channels =1# 图像的通道数，灰度图为1
path = r"C:\Users\wpx\Desktop\111.raw"
# 读取.raw文件到nparray类型
content = open(path, 'rb').read()
samples_ref = np.frombuffer(content, dtype='uint16').reshape((-1, 886, 492))
#创建一个类型为floap32的nparray类型，方便之后转换为tensor张量送入深度学习网络当中
batch_inp_np = np.zeros((1, patchsx, patchsy), dtype = 'float32')
# 将我们读取出来的raw文件内容放入到我们创建的文件当中
batch_inp_np[0, :, :] = np.float32(samples_ref[0,:, :]) * np.float32(1 / 65536)
# nparray -> torch转换类型
print("img",batch_inp_np.shape)
img_tensor = torch.from_numpy(batch_inp_np)
print("image_tensor",img_tensor.shape)

# # 利用numpy的fromfile函数读取raw文件，并指定数据格式
# img=np.fromfile(path, dtype='uint16')
# print("img",img.shape)
# # 利用numpy中array的reshape函数将读取到的数据进行重新排列。
# img=img.reshape(rows, cols, channels)
#
# # 展示图像
# cv2.imshow('Infared image-886*492-16bit',img)
# # 如果是uint16的数据请先转成uint8。不然的话，显示会出现问题。
# cv2.waitKey()
# cv2.destroyAllWindows()
# print('ok')

