import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import os.path as osp
import numpy as np
from numpy import random
from numpy.random import uniform
from util import IoU
import PIL
from PIL import Image
from random import shuffle
from train import load_net


# import random

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


class InplaceDataset(data.Dataset):
    def __init__(self, img_face_landmark, img_faces, cropsize, pnet=None, rnet=None, ratio=(2, 1, 1, 1)):
        """
        :param train_data_list: [train_data_num,[img_path,labels,[offsets],[landmark]]
        :return:
        """
        # self.img_faces = img_faces + img_face_landmark
        self.img_faces = img_face_landmark + img_faces
        shuffle(self.img_faces)
        self.crop_size = cropsize
        self.pnet = pnet
        self.rnet = rnet
        ratio_sum = float(sum(ratio))
        self.ratio = [i / ratio_sum for i in ratio]
        self.cache = []
        print('===> data set size:{}'.format(self.__len__()))
        # self.ratio = ratio
        # self.dict = {'p': 0.0, 'pf': 0.0, 'l': 1.0, 'n': 0.0}

    def get_img_faces_ldmk(self, index):
        def load_img(img_path):
            try:
                # print('===> loading the img...')
                img = Image.open(img_path)
                img = img.convert('RGB')
            except Exception:
                print('*** warning loading fail!')
                return
            return img

        img_face = self.img_faces[index]
        img_path = img_face[0]
        faces = np.array(img_face[1])
        # print('faces.ndim:{}'.format(faces.ndim))
        if faces.ndim is 1:
            # img_face_landmark
            # [absolute_img_path,[x1,x2,y1,y2],(x,y)of[left_eye,right_eye,nose,mouse_left, mouse_right]]
            faces = np.expand_dims(faces, 0)
            faces[:, :] = faces[:, (0, 2, 1, 3)]
        else:
            # [img_num * [absolute_img_path, [faces_num * 4(which is x1, y1, w, h)]]]
            faces[:, 2] += faces[:, 0]
            faces[:, 3] += faces[:, 1]
        # print('faces:{}'.format(faces))
        ldmk = None if len(img_face) < 3 else [int(i) for i in img_face[2]]

        return load_img(img_path), faces, ldmk

    def get_crop_img_label_offset_ldmk(self, img, faces, ldmk, index):
        def get_crop_img(img_np, crop_box, crop_size):
            # print('img_np:{}, crop_box:{}'.format(img_np, crop_box))
            # print('img_np.shape:{}'.format(img_np.shape))
            crop_box = [int(i) for i in crop_box]
            crop_img_np = img_np[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], :]
            # print('crop_img_np size:{}'.format(crop_img_np.shape))
            crop_img = Image.fromarray(crop_img_np, mode='RGB')
            # print('crop_img size:{}'.format(crop_img.size))
            crop_img = crop_img.resize((crop_size, crop_size), resample=PIL.Image.BILINEAR)
            return crop_img

        def get_real_label(label):
            return {'n': 'n', 'np': 'n', 'pf': 'pf' if ldmk is None else 'l',
                    'p': 'p' if ldmk is None else 'l'}.get(label)

        def cal_offset(face, box):
            if box is None:
                return []
            offset = [
                (face[0] - box[0]) / float(box[2] - box[0]),
                (face[1] - box[1]) / float(box[3] - box[1]),
                (face[2] - box[2]) / float(box[2] - box[0]),
                (face[3] - box[3]) / float(box[3] - box[1]),
            ]
            return offset

        def cal_landmark_offset(box, ldmk):
            if ldmk is None or box is None:
                return []
            else:
                minx, miny = box[0], box[1]
                w, h = box[2] - box[0], box[3] - box[1]
                ldmk_offset = [(ldmk[i] - [minx, miny][i % 2]) / float([w, h][i % 2]) for i in range(len(ldmk))]
                # print('box:{},ldmk:{},ldmk_offset:{}'.format(box, ldmk, ldmk_offset))
                return ldmk_offset

        img_np = np.array(img)
        width, height = img.size
        # random.choice(['n', 'n', 'pf', 'p'], self.ratio)
        # chose face
        if self.pnet is None:
            # negative, negative partial, partial face, positive
            label = random.choice(['n', 'np', 'pf', 'p'], p=self.ratio)
            # label = 'np'
            # print('label:{}'.format(label))
            iou_th = {'n': (0, 0.3), 'np': (0, 0.3), 'pf': (0.4, 0.65), 'p': (0.65, 1.0)}.get(label)
            sigma = {'n': 1, 'np': 0.3, 'pf': 0.1, 'p': 0.02}.get(label)
            face, face_max_size = None, None
            for i in range(10):
                face = faces[random.randint(len(faces))]
                face_max_size = max(face[2] - face[0], face[3] - face[1])
                if face_max_size > self.crop_size:
                    break
            crop_img = None
            crop_box = None
            for i in range(10):
                # if ct >= sample_num: break
                max_size = min(width, height)
                size = (uniform(-1.0, 1.0) * sigma + 1) * face_max_size
                # 保证大于剪切的尺寸要大于一个值
                size = min(max(self.crop_size, size), max_size)
                # print('size:', size)
                x1, y1 = face[0], face[1]
                crop_x1, crop_y1 = (uniform(-1.0, 1.0) * sigma + 1) * x1, (uniform(-1.0, 1.0) * sigma + 1) * y1
                crop_x1, crop_y1 = min(max(0, crop_x1), width - size), min(max(0, crop_y1), height - size)
                crop_box = np.array([int(crop_x1), int(crop_y1), int(crop_x1 + size), int(crop_y1 + size)])
                # print('crop_box:', crop_box)
                # print('faces_two_points:', faces_two_points)
                iou = IoU(crop_box, np.array([face]))
                iou_max_idx = iou.argmax()
                iou = iou.max()
                # print('iou', iou)
                # iou值不符则跳过
                if iou < iou_th[0] or iou > iou_th[1]:
                    continue
                else:
                    # print('img_np:{}'.format(img_np))
                    crop_img = get_crop_img(img_np, crop_box, self.crop_size)
                    # crop_img.show()
                    break
            return crop_img, get_real_label(label), cal_offset(face, crop_box), cal_landmark_offset(crop_box, ldmk)
        else:
            # negative, negative partial, partial face, positive
            # label = random.choice(['n', 'np', 'pf', 'p'], p=self.ratio)
            # label = 'np'
            # print('label:{}'.format(label))
            if len(self.cache) != 0:
                self.img_faces.append(self.img_faces[index])
                return self.cache.pop(0)
            iou_th = {'n': (0, 0.3), 'pf': (0.4, 0.65), 'p': (0.65, 1.0)}
            # sigma = {'n': 1, 'np': 0.3, 'pf': 0.1, 'p': 0.02}
            from detector import pnet_boxes, rnet_boxes
            bounding_boxes = pnet_boxes(img, self.pnet, show_boxes=False)
            if bounding_boxes is None:
                return None, None, None, None
            if self.rnet is not None:
                bounding_boxes_rnet = rnet_boxes(img, self.rnet, bounding_boxes, show_boxes=False)
                if len(bounding_boxes_rnet) != 0:
                    bounding_boxes = np.vstack((bounding_boxes, bounding_boxes_rnet))
            crop_img = None
            crop_box = None
            closet_face = None
            for id, box in enumerate(bounding_boxes, start=1):
                box = [min(max(0, int(box[i])), width if i % 2 == 0 else height) for i in range(4)]
                if box[2] - box[0] < self.crop_size: continue
                iou = IoU(box, faces)
                iou_max = iou.max()
                iou_index = iou.argmax()
                closet_face = faces[iou_index]
                # print('iou_max:{}, iou_index:{}'.format(iou_max, iou_index))
                # ioumax = max(iou, iou_max)
                crop_img = get_crop_img(img_np=img_np, crop_box=box, crop_size=self.crop_size)
                # img_box.show()
                # [(0, 0.3), (0.4, 0.65), (0.65, 1.0)]
                for temp_label in iou_th:
                    if iou_max < iou_th[temp_label][0] or iou_max > iou_th[temp_label][1]:
                        continue
                    else:
                        label = temp_label
                        crop_box = box
                        crop_img = get_crop_img(img_np, box, self.crop_size)
                        self.cache.append((crop_img, get_real_label(label),
                                           cal_offset(closet_face, crop_box), cal_landmark_offset(crop_box, ldmk)))

            return (None, None, None, None) if len(self.cache) == 0 else self.cache.pop(0)

    def __getitem__(self, index):
        img, faces, ldmk = self.get_img_faces_ldmk(index)
        crop_img, label, offset, ldmk = self.get_crop_img_label_offset_ldmk(img, faces, ldmk, index)
        if crop_img is None: return self.__getitem__(random.randint(0, self.__len__()))
        img_tensor = transforms.ToTensor()(crop_img)
        # label = torch.tensor([1.0]) if item[1] in ['p', 'pf', 'l'] else torch.tensor([0.0])
        landmark_flag = torch.FloatTensor([1.0 if label == 'l' else 0.0])
        label = torch.FloatTensor([1.0 if label in ['p', 'pf', 'l'] else 0.0])
        offset = torch.FloatTensor(offset if 4 == len(offset) else 4 * [0.0])
        landmark = torch.FloatTensor(ldmk if 10 == len(ldmk) else 10 * [0.0])
        # print('label type:', label.type())
        # print('data_imformation:', label, offset, landmark_flag, landmark)
        return (img_tensor, label, offset, landmark_flag, landmark)

    def __len__(self):
        # self.ct += 1
        # return self.ct
        return len(self.img_faces)


if __name__ == '__main__':
    from create_dataset import create_pnet_data_txt_parser, landmark_dataset_txt_parser, dataset_config

    args = dataset_config()
    img_faces = create_pnet_data_txt_parser(args.class_data_txt_path, args.class_data_dir)
    img_face_landmark = landmark_dataset_txt_parser(args.landmark_data_txt_path, args.landmark_data_dir)
    IDS = InplaceDataset(img_face_landmark, img_faces, cropsize=48,
                         pnet=load_net(args, 'pnet'), rnet=load_net(args, 'rnet'))
    for i, (img_tensor, label, offset, landmark_flag, landmark) in enumerate(IDS):
        print(label, offset, landmark_flag, landmark)
        print(i)
        pass
