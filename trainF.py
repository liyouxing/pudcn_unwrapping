""" @ Time: 2024/12/18 16:12  @ Author: Youxing Li  @ Email: 940756344@qq.com """
import torch
import os
import time
from datetime import datetime
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset import PUSynDataset
from Utils.accessUtils import batch_rmse_ts4, all_batch_avg_scores
from loss.CharbonnierLoss import CharbonnierLoss, EdgeLoss


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_netInfo(net, log_dir):
    num_params = 0
    log_txt = open(log_dir + "/net_info.txt", "a+")

    for index, params in enumerate(net.parameters()):
        num_params += params.numel()
        log_txt.write("index:{:0>4}\tparams:{:0>7}\n".format(index, params.numel()))
    log_txt.write("Total Params:{:0>8}\ttime:{}\n".format(
        num_params, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    log_txt.close()
    print('Total params: %d' % num_params)


def print_trainLog(log_dir, logs):
    # print log
    print(logs)
    # record log
    with open('./{}/training_log.txt'.format(log_dir), 'a') as f:
        print(logs, file=f)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialization methods provided by CycleGAN."""

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def train_network(net, loader, optimizer, loss_f1, loss_f2):
    net.train()
    train_loss = AverageMeter()

    for batch_id, (inp, gt) in enumerate(loader):
        inp, gt = inp.cuda(non_blocking=True), gt.cuda(non_blocking=True)

        pred = net(inp)
        loss = loss_f1(pred, gt) + loss_f2(pred, gt)

        train_loss.update(loss.item(), pred.size(0))  # recording
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss.avg  # average loss for each train epoch


def valid_network(net, loader):
    net.eval()
    val_rmse = AverageMeter()
    with torch.no_grad():
        for batch_id, (inp, gt, _) in enumerate(loader):
            inp, gt = inp.cuda(non_blocking=True), gt.cuda(non_blocking=True)

            pred = net(inp)
            # performance
            val_rmse.update(batch_rmse_ts4(pred, gt), pred.size(0))

    return val_rmse.avg


def main_setup_and_train(args, data_args, network):
    # setup envir
    plt.switch_backend('agg')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    torch.cuda.set_device("cuda:0" if torch.cuda.is_available() else "cpu")

    # training log
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    print_trainLog(args.log_dir, logs='-- Training HyperParams --\nlr:{}\ntrain bs:{}\n'.format(
        args.lr, args.train_batch_size))

    # dataset
    train_dataset = PUSynDataset(data_dir=args.train_txt_dir,
                                 txt_files=[args.train_inp_txt, args.train_gt_txt],
                                 data_args=data_args, isTraining=args.isTraining)
    valid_dataset = PUSynDataset(data_dir=args.val_txt_dir,
                                 txt_files=[args.val_inp_txt, args.val_gt_txt],
                                 data_args=data_args)
    # dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size,
                              shuffle=args.shuffle, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.val_batch_size, shuffle=args.shuffle,
                              num_workers=args.num_workers)
    # network
    net = network().cuda()
    net = torch.nn.DataParallel(net)
    init_weights(net)
    print_netInfo(net, args.log_dir)

    # load pretrain model
    """if args.pretrain:
        checkpoint = torch.load(args.pretrain_dir)
        net.module.nn1.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
        print("---- pretrained net info ----")
        print_network(net.module.nn1, args.log_dir)"""

    # optimizer & scheduler
    optimizer = torch.optim.Adam(net.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)
    # loss
    char_loss = CharbonnierLoss()
    edge_loss = EdgeLoss()

    # loading the latest model
    latest_path = os.path.join(args.log_dir, "net_params_latest.tar")
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path)
        initial_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        try:
            net.module.load_state_dict({k.replace(
                'module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
        except FileNotFoundError:
            print("FileNotFoundError")
        print('resuming by loading training epoch %d' % initial_epoch)
        print('continue training ... start in %d epoch' % (initial_epoch + 1))
        print('lr == %f' % scheduler.get_last_lr()[0])

    else:
        initial_epoch = 0

    best_rmse = 99.
    for epoch in range(initial_epoch, args.epochs):
        time_start = time.time()
        train_loss = train_network(net, train_loader, optimizer, char_loss, edge_loss)
        val_rmse = valid_network(net, valid_loader)
        one_epoch_time = time.time() - time_start
        scheduler.step()

        # recording log info
        print_trainLog(
            args.log_dir,
            logs='Date:{0} Epoch_Cost:{1:.0f}s Epoch:[{2}/{3}] lr:{4:.6f} Loss:{5:.4f} Val_rmse:{6:.4f}'.format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), one_epoch_time, epoch + 1, args.epochs,
                scheduler.get_last_lr()[0], train_loss, val_rmse))

        writer.add_scalars('train_loss', {'loss': train_loss}, epoch)
        writer.add_scalars('learning_rate', {'lr': scheduler.get_last_lr()[0]}, epoch)

        # saving the best model
        if val_rmse <= best_rmse:
            best_rmse = val_rmse
            save_path = os.path.join(args.log_dir, "net_params_best.tar")
            torch.save({'epoch': epoch,
                        'model_state_dict': net.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()}, save_path)

        # saving the latest model
        if epoch % args.save_epoch_interval == 0:
            save_path = os.path.join(args.log_dir, "net_params_latest.tar")
            torch.save({'epoch': epoch,
                        'model_state_dict': net.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()}, save_path)
    print("Finished Training")


if __name__ == "__main__":
    pass
