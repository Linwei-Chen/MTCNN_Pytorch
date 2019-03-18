import os
from os import path as osp
import argparse
import numpy as np
from numpy.random import uniform
import PIL
from PIL import Image
from config import DEBUG
from tqdm import tqdm
from util import IoU, convert_to_square


def config():
    parser = argparse.ArgumentParser(description='config the source data path')
    parser.add_argument('--class_data_txt_path',
                        default='/Users/chenlinwei/Dataset/WILDER_FACE/wider_face_split/wider_face_train_short.txt',
                        type=str, help='the path of WILDER FACE .txt file')
    parser.add_argument('--class_data_dir', default='/Users/chenlinwei/Dataset/WILDER_FACE/WIDER_train',
                        type=str, help='the dir of WILDER FACE image file')
    parser.add_argument('--landmark_data_txt_path',
                        default='/Users/chenlinwei/Dataset/CNN_FacePoint/train/trainImageList.txt',
                        type=str, help='the path of CelebA .txt file')
    parser.add_argument('--landmark_data_dir', default='/Users/chenlinwei/Dataset/CNN_FacePoint/train', type=str,
                        help='the dir of CelebA image file')
    parser.add_argument('--output_path', default='/Users/chenlinwei/Dataset', type=str,
                        help='the path to save the created dataset at')
    args = parser.parse_args()
    return args


def class_dataset_txt_parser(txt_path, img_dir):
    """
    :param txt_path: the path of wider_face_train_bbx_gt.txt
    :param img_dir: tha dir of WILDER_FACE/WIDER_train
    :return: img_faces type is list, shape is [img_num,[absolute_img_path,[faces_num*4(which is x1,y1,w,h)]]]
    """
    if osp.exists(txt_path):
        # *** img_faces shape :[img_path,[faces_num, 4]]
        img_faces = []
        with open(txt_path, 'r') as f:
            l = []
            lines = list(map(lambda line: line.strip().split('\n'), f))
            # lines[[str],[str],[]...]
            lines = [i[0] for i in lines]
            # lines [str,str...]
            line_counter = 0

            while line_counter < len(lines):
                img_path = lines[line_counter]
                faces_pos = []
                faces_num = int(lines[line_counter + 1])
                for i in range(faces_num):
                    face_pos = lines[line_counter + 1 + i + 1].split()
                    # [x1, y1, w, h]
                    face_pos = face_pos[:4]
                    face_pos = [int(i) for i in face_pos]
                    if DEBUG: print('face_pos:', face_pos)
                    faces_pos.append(face_pos)
                real_img_path = osp.join(img_dir, img_path)
                # if DEBUG: print(real_img_path)
                # if DEBUG: print(osp.exists(real_img_path), Image.open(real_img_path).verify())
                if osp.exists(real_img_path):
                    try:
                        Image.open(real_img_path).verify()
                        img_faces.append([real_img_path, faces_pos])
                        if DEBUG: print('Valid image')
                    except Exception:
                        if DEBUG: print('Invalid image')
                else:
                    print("*** warning:image path invalid")
                line_counter += (2 + faces_num)

        if DEBUG:
            for i in img_faces:
                print(i)
        return img_faces
    else:
        print('*** warning:WILDER_FACE txt file not exist!')


# create positive, negative, part face sample for ratio of 3:1:1 where 1 means
Augment = 5


def class_dataset(img_faces, output_path, save_dir_name, crop_size):
    save_dir = osp.join(output_path, save_dir_name)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    f = open(osp.join(save_dir, '{}.txt'.format(save_dir_name)), mode='a')
    img_id = 0
    for item in tqdm(img_faces):
        # try:
        img = Image.open(item[0])
        # get the img name, not including the file extension
        img_file_name = osp.splitext(osp.split(item[0])[1])[0]
        # if DEBUG: print(img_file_name)
        # transfer to np
        img_np = np.asarray(img)
        # get shape
        height, width, channel = img_np.shape
        # if DEBUG: print(img_np.shape)
        faces = item[1]
        crop_img_save_dir = osp.join(save_dir, img_file_name)
        if not osp.exists(crop_img_save_dir):
            os.makedirs(crop_img_save_dir)
        faces = np.array(faces)
        faces_two_points = faces.copy()
        if DEBUG: print('faces.shape:', faces.shape)
        faces_two_points[:, 2] = faces[:, 0] + faces[:, 2]
        faces_two_points[:, 3] = faces[:, 1] + faces[:, 3]
        # if DEBUG: print('faces_two_points:', faces_two_points)
        faces_num = len(faces)
        if DEBUG: print('faces_num:', faces_num)
        # *** create positive samples
        sigma_list = [0.02, 0.1, 0.2]
        for face in faces:
            x1, y1, w, h = face
            face_max_size = max(w, h)
            # counter for positive, negative, part face sample
            p_ct, n_ct, pf_ct = 0, 0, 0
            for i in range(Augment * 10):
                if pf_ct >= Augment * 1 and p_ct >= Augment * 1 and n_ct >= Augment * 1: break
                sigma = sigma_list[(p_ct >= Augment * 1) + (pf_ct >= Augment * 1 and p_ct >= Augment * 1)]
                max_size = min(width, height)
                size = (uniform(-1.0, 1.0) * sigma + 1) * face_max_size
                # 保证大于剪切的尺寸要大于一个值
                size = min(max(12, size), max_size)
                if DEBUG: print('size:', size)
                crop_x1, crop_y1 = (uniform(-1.0, 1.0) * sigma + 1) * x1, (uniform(-1.0, 1.0) * sigma + 1) * y1
                crop_x1, crop_y1 = min(max(0, crop_x1), width - size), min(max(0, crop_y1), height - size)
                crop_box = np.array([crop_x1, crop_y1, crop_x1 + size, crop_y1 + size])
                if DEBUG: print('crop_box:', crop_box, 'faces_two_points:', faces_two_points)
                iou = IoU(crop_box, faces_two_points)
                iou_max_idx = iou.argmax()
                iou = iou.max()
                if DEBUG: print('iou', iou)
                crop_box = [int(i) for i in crop_box]
                crop_img_np = img_np[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                crop_img = Image.fromarray(crop_img_np)
                crop_img = crop_img.resize((crop_size, crop_size), resample=PIL.Image.BILINEAR)
                real_face_pos = np.array([
                    (faces_two_points[iou_max_idx][0] - crop_box[0]) / w,
                    (faces_two_points[iou_max_idx][1] - crop_box[1]) / h,
                    (faces_two_points[iou_max_idx][2] - crop_box[2]) / w,
                    (faces_two_points[iou_max_idx][3] - crop_box[3]) / h,
                ])
                real_face_pos = [i for i in real_face_pos]
                if DEBUG: print('real_face_pos:', real_face_pos)
                img_id += 1
                if p_ct < Augment * 1 and iou >= 0.65:
                    crop_img_file_name = '{}_{:.6}.jpg'.format(img_id, iou)
                    _ = osp.join(crop_img_save_dir, crop_img_file_name)
                    crop_img.save(_, format='jpeg')
                    _ = osp.join(img_file_name, crop_img_file_name)
                    f.write(_ + ' p ' + '{} {} {} {}'.format(*real_face_pos) + '\n')
                    p_ct += 1
                elif pf_ct < Augment * 1 and (0.4 < iou < 0.65):
                    crop_img_file_name = '{}_{:.6}.jpg'.format(img_id, iou)
                    _ = osp.join(crop_img_save_dir, crop_img_file_name)
                    crop_img.save(_, format='jpeg')
                    _ = osp.join(img_file_name, crop_img_file_name)
                    f.write(_ + ' pf ' + '{} {} {} {}'.format(*real_face_pos) + '\n')
                    pf_ct += 1
                elif n_ct < Augment * 1 and iou < 0.3:
                    crop_img_file_name = '{}_{:.6}.jpg'.format(img_id, iou)
                    _ = osp.join(crop_img_save_dir, crop_img_file_name)
                    crop_img.save(_, format='jpeg')
                    _ = osp.join(img_file_name, crop_img_file_name)
                    f.write(_ + ' n' + '\n')
                    n_ct += 1
                else:
                    img_id -= 1
            n_ct = 0
            for i in range(Augment * 5):
                if n_ct >= Augment * 2: break
                size = np.random.randint(12, min(width, height))
                x1 = np.random.randint(0, width - size)
                y1 = np.random.randint(0, height - size)
                crop_box = np.array([x1, y1, x1 + size, y1 + size])
                iou = IoU(crop_box, faces_two_points).max()
                if iou < 0.3:
                    crop_box = [int(i) for i in crop_box]
                    crop_img_np = img_np[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                    crop_img = Image.fromarray(crop_img_np)
                    crop_img = crop_img.resize((crop_size, crop_size), resample=PIL.Image.BILINEAR)
                    crop_img_file_name = '{}_{:.6}.jpg'.format(img_id, iou)
                    _ = osp.join(crop_img_save_dir, crop_img_file_name)
                    crop_img.save(_, format='jpeg')
                    _ = osp.join(img_file_name, crop_img_file_name)
                    f.write(_ + ' n' + '\n')
                    n_ct += 1
                    img_id += 1
    f.close()


def landmark_dataset_txt_parser(txt_path, img_dir):
    if osp.exists(txt_path):
        # *** img_faces shape :[img_path,[faces_num, 4]]
        img_faces = []
        with open(txt_path, 'r') as f:
            l = []
            lines = list(map(lambda line: line.strip().split('\n'), f))
            # lines[[str],[str],[]...]
            lines = [i[0].split(' ') for i in lines]
            # lines [[path_str,pos_str]...]
            for line in lines:
                # 将路径中的'\'替换为'/'
                img_path = line[0].replace('\\', '/')
                faces_pos = [int(i) for i in line[1:5]]
                # 标注为 左右眼，嘴，左右嘴角
                landmark = [float(i) for i in line[5:]]
                real_img_path = osp.join(img_dir, img_path)
                # if DEBUG: print(real_img_path)
                # if DEBUG: print(osp.exists(real_img_path), Image.open(real_img_path).verify())
                if osp.exists(real_img_path):
                    try:
                        Image.open(real_img_path).verify()
                        img_faces.append([real_img_path, faces_pos, landmark])
                        if DEBUG: print('Valid image')
                    except Exception:
                        if DEBUG: print('Invalid image')
                else:
                    print("*** warning:image path invalid")

        for i in img_faces: print(i)
        return img_faces
    else:
        print('*** warning:WILDER_FACE txt file not exist!')


def landmark_dataset(landmark_faces, output_path, save_dir_name, crop_size):
    """
    :param landmark_faces: [absolute_img_path,[x1,y1,w,h],[left_eye,right_eye,nose,mouse_left, mouse_right]]
    :param output_path:path to save data
    :param save_dir_name:
    :param crop_size:
    :return:
    """
    '''
    for img_face in img_faces:
        absolute_img_path = img_face[0]
        [x1, y1, w, h] = img_face[1]
        [left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y,
         mouse_left_x, mouse_left_y, mouse_right_x, mouse_right_y] = img_face[2]
        print()
    '''
    # boxes: x1,y1,w,h
    # print('img_faces:', landmark_faces)
    # print('img_faces[:][1]:', landmark_faces[:][1])
    boxes = np.array([landmark_faces[i][1] for i in range(len(landmark_faces))])
    # boxes_two_point: x1,y1,x2,y2
    boxes_two_point = np.array([boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]]).T
    square_boxes = convert_to_square(boxes_two_point)
    print('square_boxes', square_boxes)
    landmark = np.array([landmark_faces[i][2] for i in range(len(landmark_faces))])
    offset = np.array([
        (boxes_two_point[:, 0] - square_boxes[:, 0]) / boxes[:, 2],
        (boxes_two_point[:, 1] - square_boxes[:, 1]) / boxes[:, 2],
        (boxes_two_point[:, 2] - square_boxes[:, 2]) / boxes[:, 3],
        (boxes_two_point[:, 3] - square_boxes[:, 3]) / boxes[:, 3],
    ])
    print('offset', offset)
    square_boxes_length = square_boxes[:, 2] - square_boxes[:, 0] + 1
    landmark = np.array([
        (landmark[:, i]) / square_boxes_length for i in range(len(landmark))
    ])
    print('landmark', landmark)


if __name__ == '__main__':
    print("Creating datasets...")
    args = config()
    print(args)
    img_faces = class_dataset_txt_parser(args.class_data_txt_path, args.class_data_dir)
    class_data_set_config = {'P_Net_dataset': 12,
                             'R_Net_dataset': 24,
                             'O_Net_dataset': 48}
    # for dir in class_data_set_config:
    #     class_dataset(img_faces, output_path=args.output_path, save_dir_name=dir, crop_size=data_set_config[dir])
    landmark_faces = landmark_dataset_txt_parser(args.landmark_data_txt_path, args.landmark_data_dir)
    landmark_data_set_config = {'O_Net_dataset': 48}
    for dir in landmark_data_set_config:
        landmark_dataset(landmark_faces, output_path=args.output_path, save_dir_name=dir,
                         crop_size=landmark_data_set_config[dir])
