import os
from os import path as osp
import argparse
import numpy as np
from numpy.random import uniform
import PIL
from PIL import Image
from config import DEBUG
from tqdm import tqdm
from util import IoU, convert_to_square, nms, calibrate_box, get_image_boxes, convert_to_square, show_bboxes, load_img
from detector import _generate_bboxes, run_first_stage, THRESHOLDS, NMS_THRESHOLDS, MIN_FACE_SIZE, pnet_boxes
from tqdm import tqdm


# set up the path and some config
def dataset_config():
    parser = argparse.ArgumentParser(description='config the source data path')
    parser.add_argument('--class_data_txt_path',
                        default='/Users/chenlinwei/Dataset/WILDER_FACE/wider_face_split/wider_face_train_short.txt',
                        type=str, help='the path of WILDER FACE .txt file')
    parser.add_argument('--class_data_dir', default='/Users/chenlinwei/Dataset/WILDER_FACE/WIDER_train',
                        type=str, help='the dir of WILDER FACE image file')
    parser.add_argument('--class_data_augment', default=5,
                        type=int, help='the dir of WILDER FACE image file')
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
    :return: img_faces type is list, shape is [img_num*[absolute_img_path,[faces_num*4(which is x1,y1,w,h)]]]
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


def create_rnet_data(min_face_size=20.0, thresholds=THRESHOLDS, nms_thresholds=NMS_THRESHOLDS):
    def img2tensor(img):
        from torchvision import transforms
        pass

    def get_name_from_path(img_path):
        return osp.splitext(osp.split(img_path)[1])[0]

    def make_dir(save_dir):
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

    def crop_img(img_np, crop_box, crop_size):
        # print('img_np:{}, crop_box:{}'.format(img_np, crop_box))
        # print('img_np.shape:{}'.format(img_np.shape))
        crop_img_np = img_np[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], :]
        # print('crop_img_np size:{}'.format(crop_img_np))
        crop_img = Image.fromarray(crop_img_np)
        crop_img = crop_img.resize((crop_size, crop_size), resample=PIL.Image.BILINEAR)
        return crop_img

    def limit_box(box):
        new_box = [min(max(0, int(box[i])), width if i % 2 == 0 else hight) for i in range(4)]
        return new_box

    def cal_offset(face, box):
        offset = [
            (face[0] - box[0]) / float(box[2] - box[0]),
            (face[1] - box[1]) / float(box[3] - box[1]),
            (face[2] - box[2]) / float(box[2] - box[0]),
            (face[3] - box[3]) / float(box[3] - box[1]),
        ]
        return offset

    def cal_landmark_offset(box, ldmk):
        if ldmk is None:
            return []
        else:
            minx, miny = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ldmk_offset = [(ldmk[i] - [minx, miny][i % 2]) / float([w, h][i % 2]) for i in range(len(ldmk))]
            # print('box:{},ldmk:{},ldmk_offset:{}'.format(box, ldmk, ldmk_offset))
            return ldmk_offset

    def txt_to_write(path, label, offset, ldmk_offset):
        s = ''
        s += '{} '.format(path)
        s += '{} '.format(label)
        for i in offset:
            s += '{} '.format(i)
        for i in ldmk_offset:
            s += '{} '.format(i)
        s += '\n'
        print(s)
        return s

    from train import load_net, config
    from config import DEVICE
    args = config()
    pnet = load_net(args, net_name='pnet').to(DEVICE)
    dataset_args = dataset_config()
    # [img_num*[absolute_img_path,[faces_num*4(which is x1,y1,w,h)]]]
    cls_img_faces = class_dataset_txt_parser(txt_path=dataset_args.class_data_txt_path,
                                             img_dir=dataset_args.class_data_dir)
    # [absolute_img_path,[x1,x2,y1,y2],(x,y)of[left_eye,right_eye,nose,mouse_left, mouse_right]]
    ldmk_img_faces = landmark_dataset_txt_parser(txt_path=dataset_args.landmark_data_txt_path,
                                                 img_dir=dataset_args.landmark_data_dir)
    img_faces = ldmk_img_faces + cls_img_faces
    # img_faces = cls_img_faces + ldmk_img_faces
    output_path = osp.join(dataset_args.output_path, 'R_net_dataset')
    txt_path = osp.join(output_path, 'R_net_dataset.txt')
    txt = open(txt_path, 'a')
    for img_face in tqdm(img_faces):
        # print('img_face:{}'.format(img_face))
        img_path = img_face[0]
        img_name = get_name_from_path(img_path)
        save_dir = osp.join(output_path, img_name)
        make_dir(save_dir)
        faces = np.array(img_face[1])
        # print('faces.ndim:{}'.format(faces.ndim))
        if faces.ndim is 1:
            faces = np.expand_dims(faces, 0)
            faces[:, :] = faces[:, (0, 2, 1, 3)]
        else:
            faces[:, 2] += faces[:, 0]
            faces[:, 3] += faces[:, 1]
        # print('faces:{}'.format(faces))
        ldmk = None if len(img_face) < 3 else [int(i) for i in img_face[2]]
        img = load_img(img_path)
        width, hight = img.size
        print('width:{}, hight:{}'.format(width, hight))
        img_np = np.array(img)
        # print('img_np:{}'.format(img_np))
        bounding_boxes = pnet_boxes(img, pnet, show_boxes=1)
        # print('bounding_boxes:{}'.format(bounding_boxes[:, 4]))
        # ioumax = 0.0
        for id, box in enumerate(bounding_boxes):
            # box[(4+1)float]
            # print('box:{}'.format(box))
            box = limit_box(box)
            # print('box:{},faces:{}'.format(box, faces))
            iou = IoU(box, faces)
            iou_max = iou.max()
            iou_index = iou.argmax()
            closet_face = faces[iou_index]
            print('iou_max:{}, iou_index:{}'.format(iou_max, iou_index))
            # ioumax = max(iou, iou_max)
            img_box = crop_img(img_np=img_np, crop_box=box, crop_size=24)
            # img_box.show()
            label = None
            # [(0, 0.3), (0.4, 0.65), (0.65, 1.0)]
            if iou <= 0.3:
                label = 'n'
                img_box_path = osp.join(save_dir, '{}.jpg'.format(id))
                img_box.save(img_box_path, format='jpeg')
                txt.write(txt_to_write(osp.relpath(img_box_path, osp.split(txt_path)[0]), label, [], []))
                pass
            elif 0.4 <= iou <= 0.65:
                label = 'pf' if ldmk is None else 'l'
                img_box_path = osp.join(save_dir, '{}.jpg'.format(id))
                img_box.save(img_box_path, format='jpeg')
                offset = cal_offset(closet_face, box)
                ldmk_offset = cal_landmark_offset(box, ldmk)
                txt.write(txt_to_write(osp.relpath(img_box_path, osp.split(txt_path)[0]), label, offset, ldmk_offset))
                pass
            elif 0.65 < iou:
                label = 'p' if ldmk is None else 'l'
                img_box_path = osp.join(save_dir, '{}.jpg'.format(id))
                img_box.save(img_box_path, format='jpeg')
                offset = cal_offset(closet_face, box)
                ldmk_offset = cal_landmark_offset(box, ldmk)
                txt.write(txt_to_write(osp.relpath(img_box_path, osp.split(txt_path)[0]), label, offset, ldmk_offset))
                pass
            # print('iou:{}'.format(iou))
    txt.close()
    pass


# TODO(chenlinwei)：检查offset值可能有误--的确有误，应当用pre的w，h
# create positive, negative, part face sample for ratio of 3:1:1 where 1 means augment
def class_dataset(img_faces, output_path, save_dir_name, crop_size, Augment=5):
    """
    :param img_faces: [img_path,[faces_num, 4==>(x1,y1,w,h)]]
    :param output_path:
    :param save_dir_name:
    :param crop_size:
    :return: class data set will be saved at output_path/save_dir_name,
            with .txt file at output_path/save_dir_name/save_dir_name.txt
    """
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
        # print('faces.shape:', faces.shape)
        faces_two_points[:, 2] = faces[:, 0] + faces[:, 2]
        faces_two_points[:, 3] = faces[:, 1] + faces[:, 3]
        # if DEBUG: print('faces_two_points:', faces_two_points)
        faces_num = len(faces)
        # print('faces_num:', faces_num)
        # *** create positive samples
        # face: [face_num, 4]
        for face in faces:
            x1, y1, w, h = face
            face_max_size = max(w, h)
            face_min_size = min(w, h)
            # skip the small faces
            if face_min_size < crop_size:
                continue
            # ['n','n', 'pf', 'p'] config
            for cl, (iou_th, sample_num, sigma) in enumerate(zip([(0, 0.3), (0, 0.3), (0.4, 0.65), (0.65, 1.0)],
                                                                 [Augment * 2, Augment * 1, Augment * 1, Augment * 1],
                                                                 [1, 0.3, 0.1, 0.02])):
                ct = 0
                for i in range(100):
                    if ct >= sample_num: break
                    max_size = min(width, height)
                    size = (uniform(-1.0, 1.0) * sigma + 1) * face_max_size
                    # 保证大于剪切的尺寸要大于一个值
                    size = min(max(crop_size, size), max_size)
                    # print('size:', size)
                    crop_x1, crop_y1 = (uniform(-1.0, 1.0) * sigma + 1) * x1, (uniform(-1.0, 1.0) * sigma + 1) * y1
                    crop_x1, crop_y1 = min(max(0, crop_x1), width - size), min(max(0, crop_y1), height - size)
                    crop_box = np.array([crop_x1, crop_y1, crop_x1 + size, crop_y1 + size])
                    # print('crop_box:', crop_box)
                    # print('faces_two_points:', faces_two_points)
                    iou = IoU(crop_box, faces_two_points)
                    iou_max_idx = iou.argmax()
                    iou = iou.max()
                    # print('iou', iou)
                    # iou值不符则跳过
                    if iou < iou_th[0] or iou > iou_th[1]:
                        continue
                    else:
                        ct += 1
                    # [y1:y2,x1:x2,:]
                    crop_box = [int(i) for i in crop_box]
                    crop_img_np = img_np[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], :]
                    crop_img = Image.fromarray(crop_img_np)
                    crop_img = crop_img.resize((crop_size, crop_size), resample=PIL.Image.BILINEAR)
                    # TODO:w,h 和x1,y1,x2,y2的关系？
                    offset = np.array([
                        (faces_two_points[iou_max_idx][0] - crop_box[0]) / float(crop_box[2] - crop_box[0]),
                        (faces_two_points[iou_max_idx][1] - crop_box[1]) / float(crop_box[3] - crop_box[1]),
                        (faces_two_points[iou_max_idx][2] - crop_box[2]) / float(crop_box[2] - crop_box[0]),
                        (faces_two_points[iou_max_idx][3] - crop_box[3]) / float(crop_box[3] - crop_box[1]),
                    ])
                    # real_face_pos = [i for i in real_face_pos]
                    # print('offset:', offset)
                    crop_img_file_name = '{}_{:.6}.jpg'.format(ct, iou)
                    _ = osp.join(crop_img_save_dir, crop_img_file_name)
                    crop_img.save(_, format='jpeg')
                    __ = osp.join(img_file_name, crop_img_file_name)
                    f.write(__ + ' {} '.format(['n', 'n', 'pf', 'p'][cl])
                            + ['', '{} {} {} {}'.format(*offset)][cl > 1] + '\n')
    f.close()


def landmark_dataset_txt_parser(txt_path, img_dir):
    """
    :param txt_path:
    :param img_dir:
    :return: [absolute_img_path,[x1,x2,y1,y2],(x,y)of[left_eye,right_eye,nose,mouse_left, mouse_right]]
    """
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

        # for i in img_faces: print(i)
        return img_faces
    else:
        print('*** warning:WILDER_FACE txt file not exist!')


def landmark_dataset(landmark_faces, output_path, save_dir_name, crop_size):
    """
    :param landmark_faces: list_shape[absolute_img_path,[x1,x2,y1,y2],(x,y)of[left_eye,right_eye,nose,mouse_left, mouse_right]]
    :param output_path: path to save dataset dir
    :param save_dir_name:
    :param crop_size: resize the face to crop size
    :return: save the landmark_dataset at output_path/save_dir_name/landmark,
            .txt at output_path/save_dir_name/save_dir_name/txt
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
    # boxes_two_point: x1,y1,x2,y2 [sample_num, 4]
    # CNN_face_point [x1,x2,y1,y2], 左上角为(0, 0)
    boxes_two_point = np.array([boxes[:, 0], boxes[:, 2], boxes[:, 1], boxes[:, 3]]).T
    print('boxes_two_point shape:', boxes_two_point.shape)
    print('boxes_two_point :', boxes_two_point)
    square_boxes = convert_to_square(boxes_two_point)
    print('square_boxes shape', square_boxes.shape)
    # landmark :[sample_num, 10]
    landmark = np.array([landmark_faces[i][2] for i in range(len(landmark_faces))])
    print('landmark shape:', landmark.shape)
    # square_boxes_length : [sample_num, 1]
    square_boxes_length = square_boxes[:, 2] - square_boxes[:, 0] + 1
    print('square_boxes_length shape:', square_boxes_length.shape)
    # offset : [sample_num, 4]
    offset = np.array([
        (boxes_two_point[:, 0] - square_boxes[:, 0]) / square_boxes_length,
        (boxes_two_point[:, 1] - square_boxes[:, 1]) / square_boxes_length,
        (boxes_two_point[:, 2] - square_boxes[:, 2]) / square_boxes_length,
        (boxes_two_point[:, 3] - square_boxes[:, 3]) / square_boxes_length,
    ]).T
    print('offset shape', offset.shape)
    print('landmark:', landmark)
    print('square_boxes:', square_boxes)
    print('square_boxes_length:', square_boxes_length)
    landmark = np.array([
        (landmark[:, i] - square_boxes[:, i % 2]) / square_boxes_length for i in range(landmark.shape[1])
    ]).T
    print('landmark', landmark)
    print('landmark.shape', landmark.shape)
    landmark_faces_path = [landmark_faces[i][0] for i in range(len(landmark_faces))]
    dataset_txt_save_path = osp.join(output_path, save_dir_name, save_dir_name + '.txt')
    dataset_save_path = osp.join(output_path, save_dir_name, 'landmark/')
    f = open(dataset_txt_save_path, 'a')
    if not osp.exists(dataset_save_path):
        os.mkdir(dataset_save_path)
    for img_path, sqbx, ofst, ldmk in zip(landmark_faces_path, square_boxes, offset, landmark):
        file_name = osp.split(img_path)[1]
        img = Image.open(img_path)
        img = img.convert('RGB')
        # h x w x c
        img_np_crop = np.array(img)[sqbx[1]:sqbx[3], sqbx[0]:sqbx[2], :]
        # test:
        # Image.fromarray(img_np_crop).show()
        img_resized = Image.fromarray(img_np_crop).resize((crop_size, crop_size))
        img_resized.save(osp.join(dataset_save_path, file_name))
        # TODO:TypeError: unsupported operand type(s) for +: 'int' and 'str'
        ofst_str = ''
        for s in [str(i) + ' ' for i in ofst]: ofst_str += s
        ldmk_str = ''
        for s in [str(i) + ' ' for i in ldmk]: ldmk_str += s
        f.write('landmark/' + file_name + ' l {} {}'.format(ofst_str, ldmk_str) + '\n')
    f.close()


if __name__ == '__main__':
    print("Creating datasets...")
    '''
    args = dataset_config()
    print(args)
    img_faces = class_dataset_txt_parser(args.class_data_txt_path, args.class_data_dir)
    class_data_set_config = {'P_Net_dataset': 12,
                             # 'R_Net_dataset': 24,
                             # 'O_Net_dataset': 48
                             }
    # for dir in class_data_set_config:
    #     class_dataset(img_faces, output_path=args.output_path, save_dir_name=dir, crop_size=class_data_set_config[dir])
    landmark_faces = landmark_dataset_txt_parser(args.landmark_data_txt_path, args.landmark_data_dir)
    landmark_data_set_config = {'O_Net_dataset': 48}
    for dir in landmark_data_set_config:
        landmark_dataset(landmark_faces, output_path=args.output_path, save_dir_name=dir,
                         crop_size=landmark_data_set_config[dir])
                         '''
    create_rnet_data()
