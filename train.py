import torch
import torch.nn as nn
import torch.optim as Opt
from torch.utils.data import DataLoader
import time
import argparse
import os
import os.path as osp
from model import P_Net, R_Net, O_Net, LossFn
from dataset import mtcnn_dataset
from config import DEBUG
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_training_data_path', type=str,
                        default='/Users/chenlinwei/Code/20190315MTCNN-Pytorch/P_Net_dataset.txt',
                        help='the path of .txt file including the training data path')
    parser.add_argument('--class_testing_data_path', type=str,
                        default=osp.join(osp.expanduser('~'), 'Dataset/CNN_FacePoint/train/testImageList.txt'),
                        help='The path of .txt file including the testing data path')
    parser.add_argument('--landmark_training_data_path', type=str,
                        default=osp.join(osp.expanduser('~'), 'Dataset/CNN_FacePoint/train/trainImageList.txt'),
                        help='the path of .txt file including the training data path')
    parser.add_argument('--landmark_testing_data_path', type=str,
                        default=osp.join(osp.expanduser('~'), 'Dataset/CNN_FacePoint/train/testImageList.txt'),
                        help='The path of .txt file including the testing data path')
    parser.add_argument('--save_folder', type=str,
                        default='none',
                        help='the folder of saved para and models')
    parser.add_argument('--lr', type=float,
                        default=0.001,
                        help='initial learning rate')
    parser.add_argument('--sub_epoch', type=int,
                        default=1000,
                        help='some batches make up a sub_epoch ')
    parser.add_argument('--batch_size', type=int,
                        default=16,
                        help='some batches make up a sub_epoch ')
    parser.add_argument('--num_workers', type=int,
                        default=4,
                        help='workers for loading the data')
    parser.add_argument('--half_lr_steps', type=int,
                        default=10000,
                        help='half the lr every half_lr_steps batches')

    args = parser.parse_args()

    return args


# optimizer =Opt.Adam(pnet.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
def load_pnet(args):
    # pnet, rnet, onet = P_Net(), R_Net(), O_Net()
    pnet = P_Net().to(DEVICE)
    try:
        print('===> loading the saved pnet weights...')
        pnet.load_state_dict(torch.load(args.saved_, map_location=DEVICE))
    except Exception:
        print('*** fail to load the saved pnet weights!')
    return pnet  # , rnet, onet


def load_para(save_folder, file_name='para.pkl'):
    para = None
    try:
        print('===> loading the saved parameters...')
        para = torch.load(osp.join(args.save_folder, file_name))
    except Exception:
        print('*** fail to load the saved parameters!')
        print('===> initailizing the parameters...')
        para = {
            'lr': args.lr,
            'train_data': random.shuffle(load_txt(args)),
            'optimizer_param': None
        }
        save_safely(para, dir_path=args.save_folder, file_name=file_name)
    return para


def get_dataset(args, train_data_list, data_dir):
    """
    :param data_dir
    :return: DataLoader
    """
    # para = load_para(args.saved_folder)
    # train_data_list = para[]

    dataset = mtcnn_dataset(data_list=train_data_list, data_dir=args.)
    return DataLoader(dataset, batch_size=1,
                      shuffle=True,
                      num_workers=0,
                      pin_memory=False)


def save_safely(file, dir_path, file_name):
    if osp.exists(dir_path):
        os.mkdir(dir_path)
    save_path = osp.join(dir_path, file_name)
    if osp.exists(save_path):
        temp_name = save_path + '.temp'
        torch.save(file, temp_name)
        os.remove(save_path)
        os.rename(temp_name, save_path)
    else:
        torch.save(file, save_path)


def train_pnet(args):
    pnet = load_pnet(args)
    optimizer = Opt.Adam(pnet.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    loss = LossFn(cls_factor=1, box_factor=0.5, landmark_factor=0.5)
    para = load_para(args.saved_folder)
    lr = para['lr']
    train_data_list = para['train_data'][:min(args.sub_epoch * args.batchsize, len(para['train_data']))]
    data_set = get_dataset(args, train_data_list, data_dir=args.class_training_data_path)


def load_txt(data_path):
    samples = []
    with open(data_path, 'r') as f:
        lines = list(map(lambda line: line.strip().split('\n'), f))
        # lines[[str],[str],[]...]
        lines = [i[0] for i in lines]
        # lines [str,str...]
        for line in lines:
            line = line.split()
            if DEBUG: print('line:', line)
            img_path = line[0]
            img_class = line[1]
            offset, landmark = [], []
            if img_class is 'p' or 'pf' or 'l':
                offset = [float(s) for s in line[2:6]]
            if img_class is 'l':
                landmark = [float(s) for s in line[6:]]
            sample = [img_path, img_class, offset, landmark]
            samples.append(sample)
    # print('samples:', samples)
    return samples


if __name__ == '__main__':
    args = config()
    load_txt(args)
