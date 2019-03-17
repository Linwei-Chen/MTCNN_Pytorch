import torch
from torchvision import transforms
import torch.utils.data.dataset as Dataset
from PIL import Image
import os.path as osp


class mtcnn_dataset(Dataset):
    def __int__(self, data_list, data_dir):
        """
        :param train_data_list: [train_data_num,[img_path,labels,[offsets],[landmark]]
        :return:
        """
        super(mtcnn_dataset, self).__init__()
        self.data_list = data_list
        self.data_dir = data_dir

    def __getitem__(self, index):
        item = self.data_list[index]
        img_path = osp.join(self.data_dir, item[0])
        img = Image.open(img_path, mode='RGB')
        img_tensor = transforms.ToTensor()(img)
        label = torch.tensor(1.0) if item[1] in ['p', 'pt', 'l'] else torch.tensor(0.0)
        offset = torch.Tensor(item[2])
        landmark = torch.Tensor(item[3])
        print('data_imformation:', osp.splitext(item[0])[0], label, offset, landmark)
        return img_tensor, label, offset, landmark

    def __index__(self):
        return len(self.data_list)
