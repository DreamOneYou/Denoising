import megengine as mge
import megengine.module as M
import megengine.functional as F
import argparse
from sklearn.model_selection import train_test_split
from glob import glob
import torch
from datasets.dataset import Dataset
import tqdm
import datetime
from megengine.data import DataLoader, RandomSampler
patchsz = 256
batchsz = 16
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNet')
    parser.add_argument('--train_steps', default=20000, type=int)
    parser.add_argument('--single', default=False)
    parser.add_argument('--input-channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='npy',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='npy',
                        help='mask file extension')

    parser.add_argument('--loss', default='BCEDiceLoss')
    parser.add_argument('--epochs', default=350, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=100, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--seed', default=116632, type=int,  help='nesterov')

    args = parser.parse_args()

    return args
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

def val(samples_ref):
    for i in tqdm.tqdm(range(0, len(samples_ref), batchsz)):
        i_end = min(i + batchsz, len(samples_ref))
        batch_inp = mge.tensor(np.float32(samples_ref[i:i_end, None, :, :]) * np.float32(1 / 65536))
        pred = net(batch_inp)

        pred = (pred.numpy()[:, 0, :, :] * 65536).clip(0, 65535).astype('uint16')
        fout.write(pred.tobytes())
    return loss

def train(args, train_loader, model, gm, optimizer, epoch, scheduler=None):
    model.train()
    train_loss = 0.
    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):

        input = input.cuda()
        target = target.cuda()

        outputs = model(input)
        loss = F.abs(outputs - target).mean()
        train_loss += loss.item()
        optimizer.zero_grad()
        gm.backward(loss)
        optimizer.step()

        print("loss:{}".format(loss))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss/(i+1)

if __name__ == '__main__':
    import random
    import numpy as np
    from megengine.utils.module_stats import module_stats
    args = parse_args()
    net = Predictor()

    # input_data = np.random.rand(1, 1, 256, 256).astype("float32")
    # total_stats, stats_details = module_stats(
    #     net,
    #     inputs=(input_data,),
    #     cal_params=True,
    #     cal_flops=True,
    #     logging_to_stdout=True,
    # )

    # print("params %.3fK MAC/pixel %.0f" % (
    # total_stats.param_dims / 1e3, total_stats.flops / input_data.shape[2] / input_data.shape[3]))

    print('loading data')
    if args.single:
        content = open('dataset/burst_raw/competition_train_input.0.1.bin', 'rb').read()
        samples_ref = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
        content = open('dataset/burst_raw/competition_train_gt.0.1.bin', 'rb').read()
        samples_gt = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
    else:
        # Data loading code
        img_paths = glob('/home/wpx/wpx/code/Liver_train_data/data/train_image/*')
        mask_paths = glob('/home/wpx/wpx/code/Liver_train_data/data/train_mask/*')

        samples_ref, val_img_paths, samples_gt, val_mask_paths = \
            train_test_split(img_paths, mask_paths, train_size=0.8, test_size=0.2, random_state=39)
        print("train_num:%s" % str(len(samples_ref)))
        print("val_num:%s" % str(len(val_img_paths)))
        train_dataset = Dataset(args, samples_ref, samples_gt, args.aug)
        val_dataset = Dataset(args, val_img_paths, val_mask_paths)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.b,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False)

    train_steps = args.train_steps
    opt = mge.optimizer.Adam(net.parameters(), lr=args.lr)
    gm = mge.autodiff.GradManager().attach(net.parameters())
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    losses = []

    rnd = random.Random(100)

    print('training')
    val_loss = 1.
    for epoch in range(0, train_steps):
        for g in opt.param_groups:
            g['lr'] = 2e-4 * (train_steps - epoch) / train_steps
        train_loss = train(args, train_loader, net, gm, opt, epoch)



        if epoch % 10 == 0:
            net.eval()
            loss = val(val_img_paths, val_mask_paths)
            print('epoch', epoch, 'val_loss', val_loss)
            if loss < val_loss:
                val_loss = loss
                fout = open('model_'+str(epoch), 'wb')
                save_path = r'Denoising/MegCup/model/{}/epoch{}_model.pth'.format(timestamp,epoch)
                mge.save(net.state_dict(), save_path)
                fout.close()