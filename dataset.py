""" @ Time: 2024/12/18 15:17  @ Author: Youxing Li  @ Email: 940756344@qq.com """
import random
import torch
import scipy.io as sio
from torch.utils.data import Dataset


# PU
class PUSynDataset(Dataset):
    """ PU Syn dataset, train: WP/UWP | test: WP/UWP """

    def __init__(self, data_dir=None, txt_files=None, data_args=None, isTraining=False):
        super().__init__()

        self.data_dir = data_dir
        self.txt_files = txt_files
        self.data_args = data_args
        self.isTraining = isTraining

        self.wp_ids = read_txt(self.data_dir + self.txt_files[0])
        self.uwp_ids = read_txt(self.data_dir + self.txt_files[1])

    def __getitem__(self, index):
        wp = sio.loadmat(self.data_dir + self.wp_ids[index])["wp"]
        uwp = sio.loadmat(self.data_dir + self.uwp_ids[index])["uwp"]
        save_name = self.wp_ids[index]

        # --- Transform --- #
        in_ts = torch.tensor(wp, dtype=torch.float32).unsqueeze(0)
        gt_ts = torch.tensor(uwp, dtype=torch.float32).unsqueeze(0)

        # --- Normalization --- #
        in_ts = (in_ts - self.data_args.wp_floor) / (self.data_args.wp_ceil - self.data_args.wp_floor)
        gt_ts = (gt_ts - self.data_args.uwp_floor) / (self.data_args.uwp_ceil - self.data_args.uwp_floor)

        if self.isTraining:
            if self.data_args.data_aug:
                in_ts, gt_ts = dRG_data_aug(in_ts, gt_ts)
            return in_ts, gt_ts
        else:
            return in_ts, gt_ts, save_name

    def __len__(self):
        return len(self.wp_ids)


# PU
class PURealDataset(Dataset):
    """ PU Real dataset, test: WP """

    def __init__(self, data_dir=None, txt_files=None, data_args=None, isTraining=False):
        super().__init__()

        self.data_dir = data_dir
        self.txt_files = txt_files
        self.data_args = data_args
        self.isTraining = isTraining

        assert self.isTraining is False
        self.wp_ids = read_txt(self.data_dir + self.txt_files[0])

    def __getitem__(self, index):
        wp = sio.loadmat(self.data_dir + self.wp_ids[index])["input"]
        save_name = self.wp_ids[index]

        # --- Transform --- #
        in_ts = torch.tensor(wp, dtype=torch.float32).unsqueeze(0)

        # --- Normalization --- #
        in_ts = (in_ts - self.data_args.wp_floor) / (self.data_args.wp_ceil - self.data_args.wp_floor)

        return in_ts, in_ts, save_name

    def __len__(self):
        return len(self.wp_ids)


# --- read image --- #
def read_txt(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    file_list = []
    for line in lines:
        file_list.append(line.strip())
    f.close()
    return file_list  # [1:10000]


# --- data augmentation --- #
def dRG_data_aug(inp, gt):
    """
    PU data augmentation
    """
    mode = random.randint(0, 5)

    if mode == 0:  # origin
        return inp, gt
    elif mode == 1:  # vertical flip
        return torch.flip(inp, dims=[-1]), torch.flip(gt, dims=[-1])
    elif mode == 2:  # horizontal flip
        return torch.flip(inp, dims=[-2]), torch.flip(gt, dims=[-2])
    elif mode == 3:  # rot 90
        return torch.rot90(inp, k=1, dims=[-2, -1]), torch.rot90(gt, k=1, dims=[-2, -1])
    elif mode == 4:  # rot 180
        return torch.rot90(inp, k=2, dims=[-2, -1]), torch.rot90(gt, k=2, dims=[-2, -1])
    else:  # rot 270
        return torch.rot90(inp, k=-1, dims=[-2, -1]), torch.rot90(gt, k=-1, dims=[-2, -1])
