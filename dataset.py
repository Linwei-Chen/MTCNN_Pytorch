import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import os.path as osp


# from config import *


class mtcnn_dataset(data.Dataset):
    def __init__(self, data_list, data_dir):
        """
        :param train_data_list: [train_data_num,[img_path,labels,[offsets],[landmark]]
        :return:
        """
        self.data_list = data_list
        self.data_dir = data_dir
        self.dict = {'p': 0.0, 'pf': 0.0, 'l': 1.0, 'n': 0.0}

    def __getitem__(self, index):
        item = self.data_list[index]
        img_path = osp.join(self.data_dir, item[0])
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img)
        # label = torch.tensor([1.0]) if item[1] in ['p', 'pf', 'l'] else torch.tensor([0.0])
        label = torch.FloatTensor([1.0 if item[1] in ['p', 'pf', 'l'] else 0.0])
        offset = torch.FloatTensor(item[2] if 4 == len(item[2]) else 4 * [0.0])
        landmark = torch.FloatTensor(item[3] if 10 == len(item[3]) else 10 * [0.0])
        landmark_flag = torch.FloatTensor([self.dict[item[1]]])
        # print('label type:', label.type())
        # print('data_imformation:', osp.splitext(item[0])[0], label, offset, landmark_flag, landmark)
        return (img_tensor, label, offset, landmark_flag, landmark)

    def __len__(self):
        return len(self.data_list)
