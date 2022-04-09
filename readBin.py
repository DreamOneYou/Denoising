import cv2
import numpy as np
import cv2
image = r"D:\MyData\MegStudio\burst_raw\competition_train_input.0.2.bin"
label = r"D:\MyData\MegStudio\burst_raw\competition_train_gt.0.2.bin"
content = open(image, 'rb').read()
samples_ref = np.frombuffer(content, dtype = 'uint16').reshape((-1,256,256))
content = open(label, 'rb').read()
samples_gt = np.frombuffer(content, dtype = 'uint16').reshape((-1,256,256))
a1 = samples_ref[4,...]
tar_img_32 = np.zeros((256,256), dtype='float32')
tar_img_32[:, :] = np.float32(a1[:, :]) * np.float32(1 / 65536)
# tar_img_32[:, :] = np.float32(a1[:, :])
print("max:{},min:{}".format(tar_img_32.max(), tar_img_32.min()))
cv2.imshow("image",a1)
cv2.waitKey(0)

print("samples_ref:",samples_ref.shape)
print("samples_gt:",samples_gt.shape)