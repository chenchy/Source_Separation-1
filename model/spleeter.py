import torch.nn as nn
import torch.nn.functional as F
import torch

batchNorm_momentum = 0.1

class block(nn.Module):
    def __init__(self, inp, out):
        super(block, self).__init__()

        self.conv1 = nn.Conv2d(inp, out, kernel_size=(5, 5), padding=(2, 2), stride=(2, 2))
        self.batch1 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.act(x)

        return x

class d_block(nn.Module):
    def __init__(self, inp, out):
        super(d_block, self).__init__()

        self.conv1 = nn.ConvTranspose2d(inp, out, kernel_size=(5, 5), padding=(2, 2), stride=(2, 2))
        self.batch1 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x, x_shape):

        x = self.conv1(x, x_shape)
        x = self.act(x)
        x = self.batch1(x)
        x = self.drop(x)

        return x

class Spleeter(nn.Module):
    def __init__(self):
        super(Spleeter, self, n_fft=4096, max_bin=None, input_mean=None, input_scale=None).__init__()

        self.nb_output_bins = n_fft // 2 + 1

        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        kernal = 16

        self.layer1 = block(2, kernal)
        self.layer2 = block(kernal, kernal * 2)
        self.layer3 = block(kernal * 2, kernal * 2**2)
        self.layer4 = block(kernal * 2**2, kernal * 2**3)
        self.layer5 = block(kernal * 2**3, kernal * 2**4)
        self.layer6 = block(kernal * 2**4, kernal * 2**5)

        self.dlayer6 = d_block(kernal * 2**5, kernal * 2**4)
        self.dlayer5 = d_block(kernal * 2**5, kernal * 2**3)
        self.dlayer4 = d_block(kernal * 2**4, kernal * 2**2)
        self.dlayer3 = d_block(kernal * 2**3, kernal * 2**1)
        self.dlayer2 = d_block(kernal * 2**2, kernal)
        self.dlayer1 = d_block(kernal * 2, 1)

        self.last = nn.Conv2d(1, 2, kernel_size=(4, 4), padding=(3, 3), dilation=(2, 2))

    def forward(self, x, mix):
        #x = x.permute(3, 0, 1, 2)
        nb_samples, nb_channels, nb_bins, nb_frames = x.data.shape

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)

        up1 = torch.cat((self.dlayer6(x6, x5.shape), x5), 1)
        up2 = torch.cat((self.dlayer5(up1, (x4.shape[0], x4.shape[1] * 2, x4.shape[2], x4.shape[3])), x4), 1)
        up3 = torch.cat((self.dlayer4(up2, (x3.shape[0], x3.shape[1] * 2, x3.shape[2], x3.shape[3])), x3), 1)
        up4 = torch.cat((self.dlayer3(up3, (x2.shape[0], x2.shape[1] * 2, x2.shape[2], x2.shape[3])), x2), 1)
        up5 = torch.cat((self.dlayer2(up4, (x1.shape[0], x1.shape[1] * 2, x1.shape[2], x1.shape[3])), x1), 1)
        up6 = self.dlayer1(up5, x.shape)

        oup = self.last(up6)

        oup = F.sigmoid(oup) * mix
        return oup

       