""" @ Time: 2024/12/18 16:13  @ Author: Youxing Li  @ Email: 940756344@qq.com """
from trainF import main_setup_and_train
from testF import main_setup_and_test
from option import get_train_args, get_test_args, get_data_args
from dataset import PURealDataset, PUSynDataset
from models.PUDCN import PUDCN

if __name__ == "__main__":
    # training
    model_type = "PUDCN"
    train_set = "RME512-Ph160"
    data_args = get_data_args()

    train_args = get_train_args(train_set=train_set, version=model_type)
    main_setup_and_train(args=train_args, data_args=data_args, network=PUDCN)

    # testing
    # test_set = "RME512-Noisy"
    # test_args = get_test_args(test_set=test_set, pretrain_set=train_set, version=model_type)
    # main_setup_and_test(test_args, data_args, PUSynDataset, PUDCN)
