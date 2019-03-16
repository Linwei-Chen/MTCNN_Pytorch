import os
from os import path as osp
import argparse
import numpy as np
from PIL import Image
from config import DEBUG


def config():
    parser = argparse.ArgumentParser(description='config the source data path')
    parser.add_argument('--WILDER_FACE_txt_path',
                        default='/Users/chenlinwei/Downloads/WILDER_FACE/wider_face_split/wider_face_train_bbx_gt.txt',
                        type=str, help='the path of WILDER FACE .txt file')
    parser.add_argument('--WILDER_FACE_dir', default='/Users/chenlinwei/Downloads/WILDER_FACE/WIDER_train',
                        type=str, help='the dir of WILDER FACE image file')
    parser.add_argument('--CelebA_txt_path', default='none', type=str, help='the path of CelebA .txt file')
    parser.add_argument('--CelebA_dir', default='none', type=str, help='the dir of CelebA image file')
    parser.add_argument('--output_path', default='/Users/chenlinwei/Dataset', type=str,
                        help='the path to save the created dataset at')
    args = parser.parse_args()
    return args


def WILDER_FACE_txt_parser(txt_path, img_dir):
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
                    # print(face_pos)
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
                line_counter += (2 + faces_num)

        if DEBUG:
            for i in img_faces:
                print(i)
        return img_faces
    else:
        print('*** warning:WILDER_FACE txt file not exist!')


def CelebA_txt_parser(txt_path, img_dir):
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
    else:
        print('*** warning:CelebA txt file not exist!')


def P_Net_dataset(img_faces, output_path, dir_name):
    dir = osp.join(output_path, 'P_Net_dataset')
    if not osp.exists(dir):
        os.makedirs(dir)
    if DEBUG:
        # Image.open()
        pass
    for item in img_faces:
        try:
            img = Image.open(item[0])
            # get the img name
            img_file_name = osp.split(osp.split(item[0])[1])[0]
            if DEBUG: print(img_file_name)
            img_np = np.asarray(img)
            height, width, channel = img_np.shape
            # if DEBUG: print(img_np.shape)
            faces = item[1]
            _ = osp.join(dir, img_file_name)
            if not osp.exists(_):
                os.makedirs(_)
            for face in faces:
                size = np.random.randint(12, min(width, height))
                nx = np.random.randint(0, width - size)
                ny = np.random.randint(0, height - size)
                crop_box = np.array([nx, ny, nx + size, ny + size])
                x1, y1, w, h = [int(i) for i in face]
                if DEBUG: print(x1, y1, w, h)
                face = img_np[y1:y1 + h, x1:x1 + w, :]
                face = Image.fromarray(face)
                # if DEBUG: face.show()
                # face.save()
        except Exception:
            continue


def R_Net_dataset():
    pass


def O_Net_dataset():
    pass


def O_Net_landmark_dataset():
    pass


if __name__ == '__main__':
    args = config()
    img_faces = WILDER_FACE_txt_parser(args.WILDER_FACE_txt_path, args.WILDER_FACE_dir)
    P_Net_dataset(img_faces, output_path=args.output_path)
