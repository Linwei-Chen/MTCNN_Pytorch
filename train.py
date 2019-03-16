import torch
import torch.nn as nn
import torch.optim as Opt
import time
import argparse
import os
import os.path as osp
from model import P_Net, R_Net, O_Net

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
    parser.add_argument('--p_net_saved_model_path', type=str,
                        default='none',
                        help='the path of saved p_net model weights')
    parser.add_argument('--r_net_saved_model_path', type=str,
                        default='none',
                        help='the path of saved r_net model weights')
    parser.add_argument('--o_net_saved_model_path', type=str,
                        default='none',
                        help='the path of saved o_net model weights')
    parser.add_argument('--lr', type=float,
                        default=0.001,
                        help='initial learning rate')
    parser.add_argument('--num_workers', type=int,
                        default=4,
                        help='workers for loading the data')
    parser.add_argument('--half_lr_steps', type=int,
                        default=10000,
                        help='half the lr every half_lr_steps batches')

    args = parser.parse_args()

    return args


# optimizer =Opt.Adam(pnet.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
def load_net(args):
    # pnet, rnet, onet = P_Net(), R_Net(), O_Net()
    pnet = P_Net()
    try:
        print('===> loading the saved pnet weights...')
        pnet.load_state_dict(torch.load(args.saved_model_path, map_location=DEVICE))
    except Exception:
        print('*** fail to load the saved pnet weights!')
    return pnet  # , rnet, onet


def load_txt(args):
    samples = []
    with open(args.class_training_data_path, 'r') as f:
        lines = list(map(lambda line: line.strip().split('\n'), f))
        # lines[[str],[str],[]...]
        lines = [i[0] for i in lines]
        # lines [str,str...]
        for line in lines:
            # print(lines[0].split())
            line = line.strip()
            img_path = line[0]
            img_class = line[1]
            offset = []
            if img_class is 'p' or 'pf':
                offset = [int(s) for s in line[2:]]
            sample = [img_path, img_class, offset]
            samples.append(sample)
    return samples


def load_para(args):
    para = None
    try:
        print('===> loading the saved parameters...')
        para = torch.load(args.saved_para_path)
    except Exception:
        print('*** fail to load the saved parameters!')
        print('===> initailizing the parameters...')
        para = {
            'lr': args.lr,
            'train_data': load_txt(args),
            'optimizer_param':None
        }
    return para


if __name__ == '__main__':
    args = config()
    load_txt(args)
