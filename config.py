import argparse
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

DEBUG = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.benchmark = True
    cudnn.deterministic = True
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    # torch.set_default_tensor_type('torch.DoubleTensor')


def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_training_data_path', type=str,
                        default=osp.join(osp.expanduser('~'), 'Dataset/CNN_FacePoint/train/trainImageList.txt'),
                        help='the path of .txt file including the training data path')
    parser.add_argument('--class_testing_data_path', type=str,
                        default=osp.join(osp.expanduser('~'), 'Dataset/CNN_FacePoint/train/testImageList.txt'),
                        help='The path of .txt file including the testing data path')
    parser.add_argument('--class_training_data_path', type=str,
                        default=osp.join(osp.expanduser('~'), 'Dataset/CNN_FacePoint/train/trainImageList.txt'),
                        help='the path of .txt file including the training data path')
    parser.add_argument('--class_testing_data_path', type=str,
                        default=osp.join(osp.expanduser('~'), 'Dataset/CNN_FacePoint/train/testImageList.txt'),
                        help='The path of .txt file including the testing data path')
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


if __name__ == '__main__':
    args_config()
