""" @ Time: 2024/12/18 16:12  @ Author: Youxing Li  @ Email: 940756344@qq.com """
import os
import torch
import torchvision.utils as utils
from torch.utils.data import DataLoader
from Utils.matUtils import save_ts3_to_mat
from Utils.accessUtils import single_rmse_ts3, single_au_ts3, single_ssim_ts3
from Utils.basicUtils import tensor_normal


def main_setup_and_test(args, data_args, dataset, network):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    score_txt = open(args.results_dir + 'scores.txt', "a+")
    test_dataset = dataset(data_dir=args.test_txt_dir, txt_files=[args.test_inp_txt, args.test_gt_txt],
                           data_args=data_args, isTraining=args.isTraining)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers)
    # create the network
    net = network().to(device)
    # loading pretrained params
    checkpoint = torch.load(args.pretrained_dir)
    net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})

    # testing
    n = 0
    total_ssim = 0.0  # ↑↑
    total_rmse = 0.0  # ↓↓
    total_au = 0.0  # ↑↑
    net.eval()
    for batch_id, (inp, gt, fs_name) in enumerate(test_loader):
        with torch.no_grad():
            inp, gt = inp.to(device), gt.to(device)
            pred, _, _ = net(inp)

        # de-normalization
        pred = pred * (data_args.uwp_ceil - data_args.uwp_floor) + data_args.uwp_floor
        gt = gt * (data_args.uwp_ceil - data_args.uwp_floor) + data_args.uwp_floor

        for ind in range(args.test_batch_size):
            f_name = fs_name[ind].split('/')[-1]
            n += 1
            print(f_name, n)
            save_dir1 = args.results_dir + 'uwp/'
            save_dir2 = args.results_dir + 'bem/'

            if not os.path.exists(save_dir1):
                os.makedirs(save_dir1)
            if not os.path.exists(save_dir2):
                os.makedirs(save_dir2)

            # accessing scores
            rmse = single_rmse_ts3(pred[ind], gt[ind])
            ssim = single_ssim_ts3(pred[ind], gt[ind], data_range=data_args.uwp_ceil)
            au, bem = single_au_ts3(pred[ind], gt[ind], bem_bool=True)
            total_ssim += + ssim
            total_rmse += rmse
            total_au += au
            print("ids:{} rmse:{:.4f} ssim:{:.4f} au:{:.4f}".format(f_name[:], rmse, ssim, au))
            score_txt.write("ids:{} rmse:{:.4f} ssim:{:.4f} au:{:.4f}\n".format(f_name[:], rmse, ssim, au))

            # save result
            save_ts3_to_mat(pred[ind], save_dir1, f_name, key='output')
            utils.save_image(tensor_normal(bem[ind]), save_dir2 + '{}'.format(f_name[:-4] + '.bmp'))

    # avg access
    avg_rmse = total_rmse / n
    avg_ssim = total_ssim / n
    avg_au = total_au / n
    print("Total info:\nMean rmse:{:.4f} Mean ssim:{:.4f} Mean au:{:.4f}".format(avg_rmse, avg_ssim, avg_au))
    score_txt.write("Total info:\nMean rmse:{:.4f} Mean ssim:{:.4f} Mean au:{:.4f}\n".format(
        avg_rmse, avg_ssim, avg_au))
    score_txt.close()


if __name__ == "__main__":
    pass
