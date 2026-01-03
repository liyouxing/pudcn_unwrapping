import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info


class ResBlock(nn.Module):
    def __init__(self, chs):
        super(ResBlock, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(chs, chs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(chs),
            nn.GELU(),
            nn.Conv2d(chs, chs, 3, 1, 1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(chs),
            nn.GELU(),
            nn.Conv2d(chs, chs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(chs),
        )

    def forward(self, x):
        oup = F.gelu(self.branch1(x) + x)

        return oup


class PUDCN(torch.nn.Module):

    def __init__(self, in_chs=1, chs=16, out_chs=1):
        super(PUDCN, self).__init__()

        self.init_block = nn.Sequential(
            nn.Conv2d(in_chs, chs, 3, 1, 1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(chs),
            nn.GELU(),
            ResBlock(chs)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(chs, chs * 2, 3, 2, 1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(chs * 2),
            nn.GELU(),
            ResBlock(chs * 2)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(chs * 2, chs * 4, 3, 2, 1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(chs * 4),
            nn.GELU(),
            ResBlock(chs * 4)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(chs * 4, chs * 8, 3, 2, 1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(chs * 8),
            nn.GELU()
        )

        self.block = nn.Sequential(
            ResBlock(chs * 8),
            ResBlock(chs * 8),
            ResBlock(chs * 8),
            ResBlock(chs * 8),
            ResBlock(chs * 8),
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(chs * 8, chs * 4, 3, 2, 1, output_padding=1, bias=False),
            nn.BatchNorm2d(chs * 4),
            nn.GELU()
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(chs * 8, chs * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(chs * 4),
            nn.GELU(),
            ResBlock(chs * 4),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(chs * 4, chs * 2, 3, 2, 1, output_padding=1, bias=False),
            nn.BatchNorm2d(chs * 2),
            nn.GELU()
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(chs * 4, chs * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(chs * 2),
            nn.GELU(),
            ResBlock(chs * 2),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(chs * 2, chs, 3, 2, 1, output_padding=1, bias=False),
            nn.BatchNorm2d(chs),
            nn.GELU()
        )

        self.coarse_block = nn.Sequential(
            nn.Conv2d(chs * 2, chs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(chs),
            nn.GELU(),
            ResBlock(chs)
        )

        self.coarse_end = nn.Sequential(
            nn.Conv2d(chs, out_chs, 3, 1, 1, padding_mode='replicate'),
            nn.Sigmoid()
        )

        self.fine_up4 = nn.Sequential(
            nn.ConvTranspose2d(chs * 8, chs, 8, 8, 0, bias=False),
            nn.BatchNorm2d(chs),
            nn.GELU()
        )
        self.fine_up3 = nn.Sequential(
            nn.ConvTranspose2d(chs * 4, chs, 4, 4, 0, bias=False),
            nn.BatchNorm2d(chs),
            nn.GELU()
        )
        self.fine_up2 = nn.Sequential(
            nn.ConvTranspose2d(chs * 2, chs, 2, 2, 0, bias=False),
            nn.BatchNorm2d(chs),
            nn.GELU()
        )

        self.fine_block = ResBlock(chs * 4)

        self.fine_end = nn.Sequential(
            nn.Conv2d(chs * 4, out_chs, 3, 1, 1, padding_mode='replicate'),
            nn.Sigmoid()
        )

    def forward(self, x):
        # coarse encoding
        x1 = self.init_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        y4 = self.block(x4)

        # coarse decoding
        y3 = self.dec3(torch.cat([self.up3(y4), x3], 1))
        y2 = self.dec2(torch.cat([self.up2(y3), x2], 1))
        y1 = self.coarse_block(torch.cat([self.up1(y2), x1], 1))
        coarse_out = self.coarse_end(y1)

        # fine stage
        fuse_feats = torch.cat([y1, self.fine_up2(y2), self.fine_up3(y3), self.fine_up4(y4)], 1)
        fine_out = self.fine_end(self.fine_block(fuse_feats))

        oup = 0.5 * (coarse_out + fine_out)

        return oup, fine_out, coarse_out  # uwp, fuw_feats, cuw_feats


if __name__ == "__main__":
    mode = 3
    net = PUDCN()
    if mode == 1:
        # real times on RTX3090
        os.environ['CUDA_VISIBLE_DEVICES'] = "1"
        net = net.cuda()
        total = 0.
        ts = torch.ones([1, 1, 256, 256]).cuda()
        for _ in range(100):
            net.eval()
            torch.cuda.synchronize()
            start = time.time()
            result = net(ts)
            torch.cuda.synchronize()
            end = time.time()
            print(end - start)
            total += (end - start)
        print("avg:" + str(total / 100.))
    elif mode == 2:
        num_params = 0
        for k, v in net.named_parameters():
            num_params += v.numel()
        print(num_params)
    else:
        # FLOPs
        macs, params = get_model_complexity_info(net, (1, 256, 256), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
