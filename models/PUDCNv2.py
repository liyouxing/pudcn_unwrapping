import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info


class PFDEM(nn.Module):
    def __init__(self, chs):
        super(PFDEM, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(chs, chs // 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(chs // 4),
            nn.GELU(),
            nn.Conv2d(chs // 4, chs // 4, 3, 1, 1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(chs // 4),
            nn.GELU(),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(chs, chs // 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(chs // 4),
            nn.GELU(),
            nn.Conv2d(chs // 4, chs // 4, 3, 1, 1, dilation=1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(chs // 4),
            nn.GELU(),
            nn.Conv2d(chs // 4, chs // 4, 3, 1, 1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(chs // 4),
            nn.GELU(),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(chs, chs // 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(chs // 4),
            nn.GELU(),
            nn.Conv2d(chs // 4, chs // 4, 3, 1, 2, dilation=2, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(chs // 4),
            nn.GELU(),
            nn.Conv2d(chs // 4, chs // 4, 3, 1, 1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(chs // 4),
            nn.GELU(),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(chs, chs // 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(chs // 4),
            nn.GELU(),
            nn.Conv2d(chs // 4, chs // 4, 3, 1, 3, dilation=3, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(chs // 4),
            nn.GELU(),
            nn.Conv2d(chs // 4, chs // 4, 3, 1, 1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(chs // 4),
            nn.GELU(),
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(chs, chs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(chs),
        )

    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        f4 = self.branch4(x)

        feats = self.conv1x1(torch.cat([f1, f2, f3, f4], 1))
        oup = F.gelu(feats + x)

        return oup


class SeLayer(nn.Module):
    def __init__(self, chs, reduction=4):
        super(SeLayer, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(chs, chs // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(chs // reduction),
            nn.GELU(),
            nn.Conv2d(chs // reduction, chs, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class FFDEM(nn.Module):
    def __init__(self, chs):
        super(FFDEM, self).__init__()
        mid_chs = chs // 16
        self.branch1 = nn.Sequential(
            nn.Conv2d(chs, mid_chs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_chs),
            nn.GELU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(chs, mid_chs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_chs),
            nn.GELU(),
            nn.Conv2d(mid_chs, mid_chs, 3, 1, 1, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(mid_chs),
            nn.GELU(),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(chs, mid_chs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_chs),
            nn.GELU(),
            nn.Conv2d(mid_chs, mid_chs, 5, 1, 2, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(mid_chs),
            nn.GELU(),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(chs, mid_chs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_chs),
            nn.GELU(),
            nn.Conv2d(mid_chs, mid_chs, 7, 1, 3, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(mid_chs),
            nn.GELU(),
        )

        self.se = SeLayer(mid_chs * 4)

        self.trans = nn.Sequential(
            nn.Conv2d(mid_chs * 4, chs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(chs),
            nn.GELU()
        )

    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        f4 = self.branch4(x)

        feats = self.trans(self.se(torch.cat([f1, f2, f3, f4], 1)))

        oup = x + feats

        return oup


class PUDCN(torch.nn.Module):

    def __init__(self, in_chs=1, chs=16, out_chs=1):
        super(PUDCN, self).__init__()

        self.init_block = nn.Sequential(
            nn.Conv2d(in_chs, chs, 3, 1, 1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(chs),
            nn.GELU(),
            PFDEM(chs)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(chs, chs * 2, 3, 2, 1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(chs * 2),
            nn.GELU(),
            PFDEM(chs * 2)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(chs * 2, chs * 4, 3, 2, 1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(chs * 4),
            nn.GELU(),
            PFDEM(chs * 4)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(chs * 4, chs * 8, 3, 2, 1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(chs * 8),
            nn.GELU()
        )

        self.block = nn.Sequential(
            PFDEM(chs * 8),
            PFDEM(chs * 8),
            PFDEM(chs * 8),
            PFDEM(chs * 8),
            PFDEM(chs * 8),
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
            PFDEM(chs * 4),
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
            PFDEM(chs * 2),
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
            PFDEM(chs)
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

        self.fine_block = FFDEM(chs * 4)

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
