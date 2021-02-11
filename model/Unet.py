import torch
import torch.nn as nn
import torch.nn.functional as F

batchNorm_momentum = 0.1

class block(nn.Module):
    def __init__(self, inp, out, ksize, pad, ds_ksize, ds_stride):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(inp,out, kernel_size=ksize, padding=pad)
        self.bn1 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.conv2 = nn.Conv2d(out, out, kernel_size=ksize, padding=pad)
        self.bn2 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.skip = nn.Conv2d(inp, out, kernel_size=1, padding=0)
        self.ds = nn.Conv2d(out, out, kernel_size=ds_ksize, stride=ds_stride, padding=0)

    def forward(self, x):
        x11 = F.leaky_relu(self.bn1(self.conv1(x)))
        x12 = F.leaky_relu(self.bn2(self.conv2(x11)))
        x12 += self.skip(x)
        xp = self.ds(x12)
        return xp, xp, x12.size()

class d_block(nn.Module):
    def __init__(self, inp, out, isLast, ksize, pad, ds_ksize, ds_stride):
        super(d_block, self).__init__()
        self.conv2d = nn.ConvTranspose2d(inp, int(inp/2), kernel_size=ksize, padding=pad)
        self.bn2d = nn.BatchNorm2d(int(inp/2), momentum= batchNorm_momentum)
        self.conv1d = nn.ConvTranspose2d(int(inp/2), out, kernel_size=ksize, padding=pad)
        if not isLast: 
            self.bn1d = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
            self.us = nn.ConvTranspose2d(inp-out, inp-out, kernel_size=ds_ksize, stride=ds_stride)
        else: self.us = nn.ConvTranspose2d(inp, inp, kernel_size=ds_ksize, stride=ds_ksize)

    def forward(self, x, idx, size, isLast, skip):
        x = self.us(x,output_size=size)
        if not isLast: x = torch.cat((x, skip), 1)
        x = self.bn2d(F.leaky_relu(self.conv2d(x)))
        if isLast: x = self.conv1d(x)
        else:  x = self.bn1d(F.leaky_relu(self.conv1d(x)))
        return x

class Encode(nn.Module):
    def __init__(self, nb_channels):
        super(Encode, self).__init__()

        f_size = 16 
        k_size = (3,3)
        p_size = (1,1)
        ds_k = (3,1)
        ds_s = (3,1)

        self.block1 = block(nb_channels,f_size,k_size,p_size,ds_k,ds_s)
        self.block2 = block(f_size,f_size*2,k_size,p_size,ds_k,ds_s)
        self.block3 = block(f_size*2,f_size*3,k_size,p_size,ds_k,ds_s)

        self.conv2 = nn.Conv2d(f_size*2,f_size*2, kernel_size=k_size, padding=p_size)
        self.conv3 = nn.Conv2d(f_size,f_size, kernel_size=k_size, padding=p_size)

    def forward(self, x):

        x1,idx1,s1 = self.block1(x) 
        x2,idx2,s2 = self.block2(x1)
        x3,idx3,s3 = self.block3(x2)
        c2=self.conv2(x2)
        c3=self.conv3(x1)

        return x3,[idx1,idx2,idx3],[s1,s2,s3],[x1,c3,c2]


class Decode(nn.Module):
    def __init__(self, nb_channels):
        super(Decode, self).__init__()
        f_size = 16
        k_size = (3,3)
        p_size = (1,1)
        ds_k = (3,1)
        ds_s = (3,1)
        self.d_block2 = d_block(f_size * 5,f_size*2,False,k_size,p_size,ds_k,ds_s)
        self.d_block3 = d_block(f_size * 3,f_size,False,k_size,p_size,ds_k,ds_s)
        self.d_block4 = d_block(f_size, nb_channels,True,k_size,p_size,ds_k,ds_s)

    def forward(self, x, idx, s, c):
        x = self.d_block2(x,idx[2],s[2],False,c[2])
        x = self.d_block3(x,idx[1],s[1],False,c[1])
        pred = self.d_block4(x,idx[0],s[0],True,c[0])
        return pred

class Unet(nn.Module):
    def __init__(self, n_fft=4096, max_bin=None, input_mean=None, input_scale=None,):
        super(Unet, self).__init__()

        self.nb_output_bins = n_fft // 2 + 1
        nb_channels = 2

        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.encode = Encode(nb_channels)
        self.decode = Decode(nb_channels)
        self.last = nn.Linear(self.nb_bins, self.nb_output_bins)

        if input_mean is not None:
            input_mean = torch.from_numpy(
                input_mean[:self.nb_bins]
            ).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(
                1.0*input_scale[:self.nb_bins]
            ).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = nn.Parameter(input_mean)
        self.input_scale = nn.Parameter(input_scale)

        self.output_scale = nn.Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        self.output_mean = nn.Parameter(
            torch.ones(self.nb_output_bins).float()
        )

    def forward(self, x, mix, emb=None):
        x = x.permute(3, 0, 1, 2)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        # crop
        x = x[..., :self.nb_bins].contiguous()

        x += self.input_mean
        x *= self.input_scale
       
        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x, idx, s, c = self.encode(x.permute(1,2,3,0).contiguous())
        if emb:
            print(x.shape, emb.shape)

        x = self.decode(x, idx, s, c)
        x = self.last(x.permute(0,1,3,2)).permute(2, 0, 1, 3).contiguous()

        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x).permute(1, 2, 3, 0) * mix
     
        return x
