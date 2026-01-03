""" @ Time: 2024/12/18 15:44  @ Author: Youxing Li  @ Email: 940756344@qq.com """
import argparse


def get_train_args(train_set, valid_set=None, version="Ours"):
    parser = argparse.ArgumentParser(description='Options for training PU Network')
    trainset = {"RME12000", "RME12000-Noisy", "RME12000-Disc", "RME12000-Mix", "RME22000", "RME512-Ph160"}
    train_dataset = train_set
    assert train_dataset in trainset, 'train dataset not exist'
    if valid_set is None:
        valid_dataset = train_dataset
    else:
        valid_dataset = valid_set
        assert valid_dataset in trainset, 'valid dataset not exist'

    train_txt_dir = "../PhUn-data/" + train_dataset + "/"
    val_txt_dir = "../PhUn-data/" + valid_dataset + "/"
    log_dir = "./log_train/" + train_dataset + "/" + version + "/"

    train_inp_txt = 'train-input.txt'
    train_gt_txt = 'train-gt.txt'

    val_inp_txt = 'valid-input.txt'
    val_gt_txt = 'valid-gt.txt'

    if train_dataset in {"RME12000", "RME12000-Noisy", "RME12000-Disc", "RME12000-Mix", "RME512-Ph160"}:  # 10000+
        epochs = 120
    elif train_dataset in {"RME22000"}:  # 20000+
        epochs = 60
    else:
        epochs = 0
        milestones = []

    save_epoch_interval = epochs // 50 if epochs >= 100 else 1

    # GPU
    parser.add_argument('--gpu_ids', default="1, 0", help='gpu for training')
    parser.add_argument('--cudnn', type=bool, default=True, help='cudnn accelerate')
    # data info
    parser.add_argument('--train_txt_dir', default=train_txt_dir, help='train data txt dir')
    parser.add_argument('--train_inp_txt', default=train_inp_txt, help='train input txt')
    parser.add_argument('--train_gt_txt', default=train_gt_txt, help='train gt txt')
    parser.add_argument('--val_txt_dir', default=val_txt_dir, help='valid data txt dir')
    parser.add_argument('--val_inp_txt', default=val_inp_txt, help='valid input txt')
    parser.add_argument('--val_gt_txt', default=val_gt_txt, help='valid gt txt')

    parser.add_argument('--log_dir', default=log_dir, help='training log dir')
    # training info
    parser.add_argument('--train_batch_size', type=int, default=18, help='train batch size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='val batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--epochs', type=int, default=epochs, help='train epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader thread number')
    parser.add_argument('--shuffle', type=bool, default=True, help='dataloader shuffle')
    parser.add_argument('--isTraining', type=bool, default=True, help='training or validating')
    parser.add_argument('--pretrain', type=bool, default=False, help='w/o pretrain?')
    parser.add_argument('--pretrain_dir', default=None, help='pretrain dir')

    # scheduler info
    # parser.add_argument('--milestones', default=milestones, help='milestones in MultiStepLR')
    # parser.add_argument('--gamma', type=float, default=0.2, help='gamma in MultiStepLR')
    parser.add_argument('--T_0', type=int, default=int(epochs // 3), help='T_0 in CosineAnnealingLR')
    parser.add_argument('--T_mult', type=int, default=2, help='times by T_0 in CosineAnnealingLR')
    parser.add_argument('--eta_min', type=float, default=10e-6, help='minimum lr in CosineAnnealingLR')

    # optimizer info
    parser.add_argument('--weight_decay', type=float, default=1e-10, help='weight_decay in Adam')
    # log info
    parser.add_argument('--print_batch_size', type=int, default=30, help='print info with batch size interval')
    parser.add_argument('--save_epoch_interval', type=int, default=save_epoch_interval,
                        help='the frequency for saving the latest model')

    args = parser.parse_args()

    return args


"""
Data Range
    RME12000-**: 0~80
    RME22000-**: 0~40
    RME512-Ph160: 0~160
"""


def get_data_args():
    parser = argparse.ArgumentParser(description='PU data option')  # create parser

    # data info
    parser.add_argument('--wp_floor', type=float, default=-3.1416, help='the floor of wrapped phase')
    parser.add_argument('--wp_ceil', type=float, default=3.1416, help='the ceil of wrapped phase')
    parser.add_argument('--uwp_floor', type=float, default=0, help='the minimum of absolute phase range')
    parser.add_argument('--uwp_ceil', type=float, default=160, help='the maximum of absolute phase range')

    # dataset settings
    parser.add_argument('--data_aug', type=bool, default=True, help='training data aug')

    args = parser.parse_args()
    return args


def get_test_args(test_set, pretrain_set, version="Ours"):
    parser = argparse.ArgumentParser(description='Options for testing PU Network')
    test_dataset = test_set
    pretrain_dataset = pretrain_set
    assert test_dataset in {"RME12000", "RME12000-Noisy", "RME12000-Disc", "RME12000-Mix", "RME1000-Discs",
                            "RME512-Ph160"}, 'dataset not exist'
    assert pretrain_dataset in {"RME12000", "RME12000-Noisy", "RME12000-Disc", "RME12000-Mix",
                                "RME512-Ph160"}, 'dataset not exist'

    test_txt_dir = '../PhUn-data/' + test_dataset + "/"
    pretrained_dir = './log_train/' + pretrain_dataset + "/" + version + '/net_params_best.tar'
    test_inp_txt = 'test-input.txt'
    test_gt_txt = 'test-gt.txt'

    if test_dataset == pretrain_dataset:
        results_dir = './results/' + test_dataset + "/" + version + "/"
    else:
        results_dir = './results/' + test_dataset + "/" + pretrain_dataset + "/" + version + "/"

    parser.add_argument('--gpu_ids', default='1', help='select gpu id for testing')
    parser.add_argument('--test_txt_dir', default=test_txt_dir, help='test data txt dir')
    parser.add_argument('--test_inp_txt', default=test_inp_txt, help='test input txt')
    parser.add_argument('--test_gt_txt', default=test_gt_txt, help='test gt txt')
    parser.add_argument('--results_dir', default=results_dir, help='test results dir')
    parser.add_argument('--pretrained_dir', default=pretrained_dir, help='load *.tar pretrained model')
    parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')
    parser.add_argument('--isTraining', type=bool, default=False, help='training or testing')
    parser.add_argument('--num_workers', type=int, default=4, help='data loading thread number')

    args = parser.parse_args()
    return args
