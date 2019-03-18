import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import os.path as osp
from config import *


class mtcnn_dataset(data.Dataset):
    def __init__(self, data_list, data_dir):
        """
        :param train_data_list: [train_data_num,[img_path,labels,[offsets],[landmark]]
        :return:
        """
        self.data_list = data_list
        self.data_dir = data_dir
        self.dict = {'p': 1.0, 'pf': 1.0, 'l': 1.0, 'n': 0.0}

    def __getitem__(self, index):
        item = self.data_list[index]
        img_path = osp.join(self.data_dir, item[0])
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img)
        label = torch.tensor([1.0]) if item[1] in ['p', 'pf', 'l'] else torch.tensor([0.0])
        offset = torch.Tensor(item[2]) if 4 == len(item[2]) else torch.tensor([0.0 for i in range(4)])
        landmark = torch.Tensor(item[3]) if 10 == len(item[3]) else torch.tensor([0.0 for i in range(10)])
        # print('data_imformation:', osp.splitext(item[0])[0], label, offset, landmark)
        return (img_tensor, label, offset, landmark), torch.tensor([self.dict[item[1]]])

    def __len__(self):
        return len(self.data_list)
