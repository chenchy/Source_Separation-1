import torch.nn as nn
import torch.nn.functional as F
import torch

batchNorm_momentum = 0.1
kernal = 16


class block(nn.Module):
    def __init__(self, inp, out):
        super(block, self).__init__()

        self.conv1 = nn.Conv2d(inp, out, kernel_size=(5, 5), padding=(2, 2)) #, stride=(2, 2))
        self.batch1 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.act = nn.LeakyReLU()

        self.ds = nn.Conv2d(out, out, kernel_size=(2, 2), stride=(2, 2))
        self.ds_bn = nn.BatchNorm2d(out, momentum= batchNorm_momentum)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.act(x)

        x = self.ds_bn(self.ds(x))

        return x

class d_block(nn.Module):
    def __init__(self, inp, out):
        super(d_block, self).__init__()

        self.conv1 = nn.Conv2d(out, out, kernel_size=(5, 5), padding=(2, 2)) #, stride=(2, 2))
        self.batch1 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.act = nn.LeakyReLU()
        #self.drop = nn.Dropout(0.5)

        self.us = nn.ConvTranspose2d(inp, out, kernel_size=(2, 2), stride=(2, 2))
        self.us_bn = nn.BatchNorm2d(out, momentum= batchNorm_momentum)

    def forward(self, x, x_shape):

        x = self.us_bn(self.us(x, x_shape))

        x = self.conv1(x)
        x = self.act(x)
        x = self.batch1(x)
        #x = self.drop(x)

        return x

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        self.layer1 = block(2, kernal)
        self.layer2 = block(kernal, kernal * 2)
        self.layer3 = block(kernal * 2, kernal * 2**2)
        #self.layer4 = block(kernal * 2**2, kernal * 2**3)
        #self.layer5 = block(kernal * 2**3, kernal * 2**4)
        #self.layer6 = block(kernal * 2**4, kernal * 2**5)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = None #self.layer4(x3)
        x5 = None #self.layer5(x4)
        x6 = None #self.layer6(x5)
        return [x1, x2, x3, x4, x5, x6]


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        #self.dlayer6 = d_block(kernal * 2**5, kernal * 2**4)
        #self.dlayer5 = d_block(kernal * 2**5, kernal * 2**3)
        #self.dlayer4 = d_block(kernal * 2**4, kernal * 2**2)
        self.dlayer3 = d_block(kernal * 2**2, kernal * 2**1)
        self.dlayer2 = d_block(kernal * 2**2, kernal)
        self.dlayer1 = d_block(kernal * 2, 2)

    def forward(self, x, x1, x2, x3, x4, x5, x6):
        #up1 = torch.cat((self.dlayer6(x6, x5.shape), x5), 1)
        #up2 = torch.cat((self.dlayer5(up1, (x4.shape[0], x4.shape[1] * 2, x4.shape[2], x4.shape[3])), x4), 1)
        #up3 = torch.cat((self.dlayer4(up2, (x3.shape[0], x3.shape[1] * 2, x3.shape[2], x3.shape[3])), x3), 1)
        up4 = torch.cat((self.dlayer3(x3, (x2.shape[0], x2.shape[1] * 2, x2.shape[2], x2.shape[3])), x2), 1)
        up5 = torch.cat((self.dlayer2(up4, (x1.shape[0], x1.shape[1] * 2, x1.shape[2], x1.shape[3])), x1), 1)
        up6 = self.dlayer1(up5, x.shape)
        return up6


class Spleeter(nn.Module):
    def __init__(self, use_emb=False):
        super(Spleeter, self).__init__()

        self.encoder_stacks = nn.ModuleList()
        for i in range(4):
            self.encoder_stacks.append(encoder())

        self.decoder = decoder()

        self.last = nn.Conv2d(1, 2, kernel_size=(4, 4), padding=(3, 3), dilation=(2, 2))
        self.use_emb = use_emb
        if self.use_emb:
            self.emb_transform = nn.Linear(16384, 128)

    def forward(self, x, mix, y):
        #x = x.permute(3, 0, 1, 2)
        nb_samples, nb_channels, nb_bins, nb_frames = x.data.shape

        n_sources = int(y.shape[0] // x.shape[0])

        output = []
        emb = []

        for s in range(n_sources):
            x1, x2, x3, x4, x5, x6 = self.encoder_stacks[s](x)
            emb.append(self.emb_transform(x3.permute(0, 3, 1, 2).reshape(-1, 16384)))
            oup = self.decoder(x, x1, x2, x3, x4, x5, x6)
            oup = F.sigmoid(oup) * mix
            output.append(oup)

        #oup = self.last(up6)
        output = torch.stack(output)
        
        output = output.permute(1, 0, 2, 3, 4).reshape(-1, output.shape[2], output.shape[3], output.shape[4])

        if self.use_emb:
            return output, emb
        else:
            return output, emb

       