import megengine as mge
import megengine.module as M
import megengine.functional as F
import random
import torch
import argparse
import numpy as np
print('prediction')
patchsz = 256
batchsz = 16
class Predictor(M.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = M.Sequential(
            M.Conv2d(4, 50, 3, padding = 1, bias = True),
            M.LeakyReLU(negative_slope = 0.125),
            M.Conv2d(50, 50, 3, padding = 1, bias = True),
            M.LeakyReLU(negative_slope = 0.125),
        )
        self.conv2 = M.Sequential(
            M.Conv2d(50, 50, 3, padding = 1, bias = True),
            M.LeakyReLU(negative_slope = 0.125),
            M.Conv2d(50, 50, 3, padding = 1, bias = True),
            M.LeakyReLU(negative_slope = 0.125),
        )
        self.conv3 = M.Sequential(
            M.Conv2d(50, 50, 3, padding = 1, bias = True),
            M.LeakyReLU(negative_slope = 0.125),
            M.Conv2d(50, 4, 3, padding = 1, bias = True),
            M.LeakyReLU(negative_slope = 0.125),
        )
    def forward(self, x):
        n, c, h, w = x.shape
        x = x.reshape((n, c, h // 2, 2, w // 2, 2)).transpose((0, 1, 3, 5, 2, 4)).reshape((n, c * 4, h // 2, w // 2))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape((n, c, 2, 2, h // 2, w // 2)).transpose((0, 1, 4, 2, 5, 3)).reshape((n, c, h, w))
        return x

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='LITS',
                        help='model name')
    parser.add_argument('--mode', default='models/LITS_UNet_lym/20220113_190502_VNet/epoch322-0.9714-0.8921_model.pth', help='')
    parser.add_argument('--pred_path', default='pred_test', help='')
    parser.add_argument('--deep_supervision', default=False, help='')
    parser.add_argument('--patchsize', default=(64, 160, 160), type=int, metavar='N',
                        help='number of slice')

    args = parser.parse_args()

    return args
if __name__=="__main__":
    args = parse_args()

    net = Predictor()
    net.load_state_dict(mge.load(args.mode))
    net.eval()
    content = open('competition_test_input.0.1.bin', 'rb').read()
    samples_ref = np.frombuffer(content, dtype = 'uint16').reshape((-1,256,256))
    fout = open('competition_prediction.0.1.bin', 'wb')

    import tqdm

    for i in tqdm.tqdm(range(0, len(samples_ref), batchsz)):
        i_end = min(i + batchsz, len(samples_ref))
        batch_inp = mge.tensor(np.float32(samples_ref[i:i_end, None, :, :]) * np.float32(1 / 65536))
        pred = net(batch_inp)
        pred = (pred.numpy()[:, 0, :, :] * 65536).clip(0, 65535).astype('uint16')
        fout.write(pred.tobytes())

    fout.close()