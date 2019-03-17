import torch
import torch.nn as nn
import torch.optim as opt
import torchvision
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
    parser.add_argument('--p_net_data', type=str,
                        default='/Users/chenlinwei/Dataset/P_Net_dataset/P_Net_dataset.txt',
                        help='the path of .txt file including the training data path')
    parser.add_argument('--r_net_data', type=str,
                        default='/Users/chenlinwei/Dataset/P_Net_dataset/R_Net_dataset.txt',
                        help='the path of .txt file including the training data path')
    parser.add_argument('--o_net_data', type=str,
                        default='/Users/chenlinwei/Dataset/O_Net_dataset/O_Net_dataset.txtt',
                        help='the path of .txt file including the training data path')
    parser.add_argument('--class_training_data_path', type=str,
                        default='/Users/chenlinwei/Code/20190315MTCNN-Pytorch/P_Net_dataset.txt',
                        help='the path of .txt file including the training data path')
    parser.add_argument('--class_testing_data_path', type=str,
                        default=osp.join(osp.expanduser('~'), 'Dataset/CNN_FacePoint/train/testImageList.txt'),
                        help='The path of .txt file including the testing data path')
    parser.add_argument('--landmark_training_data', type=str,
                        default=osp.join(osp.expanduser('~'), 'Dataset/CNN_FacePoint/train/trainImageList.txt'),
                        help='the path of .txt file including the training data path')
    parser.add_argument('--landmark_testing_data', type=str,
                        default=osp.join(osp.expanduser('~'), 'Dataset/CNN_FacePoint/train/testImageList.txt'),
                        help='The path of .txt file including the testing data path')
    parser.add_argument('--save_folder', type=str,
                        default='/Users/chenlinwei/Dataset/MTCNN_weighs',
                        help='the folder of p/r/onet_para.pkl, p/r/onet.pkl saved')
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
def load_net(args, net_name):
    # pnet, rnet, onet = P_Net(), R_Net(), O_Net()
    net_list = {'pnet': P_Net(), 'rnet': R_Net(), 'onet': O_Net()}
    try:
        net = net_list[net_name].to(DEVICE)
        try:
            print('===> loading the saved net weights...')
            net.load_state_dict(torch.load(osp.join(args.saved_folder, net_name + '.pkl'), map_location=DEVICE))
            return net  # , rnet, onet
        except Exception:
            print('*** fail to load the saved net weights!')
            return net
    except Exception:
        print('*** Net name wrong!')


def load_para(file_name='pnet_para.pkl'):
    para = None
    try:
        print('===> loading the saved parameters...')
        para = torch.load(osp.join(args.save_folder, file_name))
    except Exception:
        print('*** fail to load the saved parameters!')
        print('===> initailizing the parameters...')
        para = {
            'lr': args.lr,
            'iter': 0,
            # 'train_data': random.shuffle(load_txt()),
            'optimizer_param': None
        }
        save_safely(para, dir_path=args.save_folder, file_name=file_name)
    return para


def get_dataset(args, net_name):
    net_list = {'pnet': args.p_net_data,
                'rnet': args.r_net_data,
                'onet': args.o_net_data}
    _ = None
    try:
        _ = net_list[net_name]
    except Exception:
        print('*** Net name wrong!')

    # print(osp.exists(_))
    print('===> loading the data_list, path is {}'.format(_))
    data_list = load_txt(_)
    data_dir = osp.split(_)[0]
    print('===> the data dir is {}'.format(data_dir))
    dataset = mtcnn_dataset(data_list=data_list, data_dir=data_dir)
    return DataLoader(dataset,
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=args.num_workers,
                      pin_memory=False)


def save_safely(file, dir_path, file_name):
    if not osp.exists(dir_path):
        os.mkdir(dir_path)
        print('*** dir not exist, created one')
    save_path = osp.join(dir_path, file_name)
    if osp.exists(save_path):
        temp_name = save_path + '.temp'
        torch.save(file, temp_name)
        os.remove(save_path)
        os.rename(temp_name, save_path)
        print('*** find the file conflict while saving, saved safely')
    else:
        torch.save(file, save_path)


def train_net(args, net_name='pnet'):
    pnet = load_net(args, net_name)
    optimizer = opt.Adam(pnet.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    loss = LossFn(cls_factor=1, box_factor=0.5, landmark_factor=0.5)
    para = load_para()
    lr = para['lr']
    iter_count = para['iter']
    if para['optimizer_param'] is not None:
        optimizer.state_dict()['param_groups'][0].update(para['optimizer_param'])
        print('===> updated the param of optimizer.')
    data_set = get_dataset(args, net_name)
    for iter, (img_tensor, label, offset, landmark) in enumerate(data_set, iter_count):
        wrap = (img_tensor, label, offset, landmark)
        (img_tensor, label, offset, landmark) = [i.to(DEVICE) for i in wrap]
        det, box, landmark = pnet(img_tensor)
        optimizer.zero_grad()
        all_loss = loss.total_loss(gt_label=label, pred_label=det, gt_offset=offset, pred_offset=box)
        print('loss:{:.8f}'.format(all_loss.item()))
        all_loss.backward()
        optimizer.step()


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
    train_net(args,'pnet')
