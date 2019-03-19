from config import *
from train import config, load_net
from torchvision import transforms
import math
import numpy as np
from PIL import Image
import torch
from util import nms, calibrate_box, get_image_boxes, convert_to_square, show_bboxes, load_img


def detect_faces(args, img, min_face_size=20.0, thresholds=[0.6, 0.7, 0.8],
                 nms_thresholds=[0.7, 0.7, 0.7]):
    pnet, rnet, onet = load_net(args, 'pnet'), load_net(args, 'rnet'), load_net(args, 'onet')
    onet.eval()
    width, height = img.size
    min_length = min(height, width)
    print('img min_length is {}'.format(min_length))
    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)

    scales = []
    # min_face_size 哪来的？
    m = min_detection_size / min_face_size
    # 缩放原图使得最小脸尺寸为12pix
    min_length *= m
    # 将图片从最小脸为12pix到整张图为12pix，保存对应的缩放比例，都为小于1的数？
    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m * factor ** factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1
    bounding_boxes = []
    for s in scales:  # run P-Net on different scales
        boxes = run_first_stage(img, pnet, scale=s, threshold=thresholds[0])
        bounding_boxes.append(boxes)
        # bounding_boxes shape:[scales,boxes_num_each_sale,5]
    # 把每个scale找到的框框全部打开堆在一起
    # [total_boxes_num, 5] 是list
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    # print(len(bounding_boxes), len(bounding_boxes[0]))
    bounding_boxes = np.vstack(bounding_boxes)
    # print(bounding_boxes.shape)

    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]
    # 根据 w、h 对 x1,y1,x2,y2 的位置进行微调
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
    # 将检测出的框转化成矩形
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
    # print('bounding_boxes:', len(bounding_boxes), bounding_boxes)
    show_bboxes(img, bounding_boxes, []).show()

    # STAGE 2
    img_boxes = get_image_boxes(bounding_boxes, img, size=24)
    img_boxes = torch.FloatTensor(img_boxes)
    output = rnet(img_boxes)
    probs = output[0].data.numpy()  # shape [n_boxes, 1]
    offsets = output[1].data.numpy()  # shape [n_boxes, 4]

    keep = np.where(probs[:, 0] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 0].reshape((-1,))
    offsets = offsets[keep]

    keep = nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
    show_bboxes(img, bounding_boxes, []).show()

    # STAGE 3
    img_boxes = get_image_boxes(bounding_boxes, img, size=48)
    if len(img_boxes) == 0:
        return [], []
    img_boxes = torch.FloatTensor(img_boxes)
    output = onet(img_boxes)

    probs = output[0].data.numpy()  # shape [n_boxes, 1]
    offsets = output[1].data.numpy()  # shape [n_boxes, 4]
    landmarks = output[2].data.numpy()  # shape [n_boxes, 10]

    keep = np.where(probs[:, 0] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    # 用更大模型的置信度对原置信度进行更新
    bounding_boxes[:, 4] = probs[keep, 0].reshape((-1,))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # compute landmark points
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
    # landmark[,前5个为x，后5个为y]
    # 在左上角坐标的基础上，通过 w，h 确定脸各关键点的坐标。
    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0::2]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 1::2]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]
    show_bboxes(img, bounding_boxes, landmarks).show()
    return bounding_boxes, landmarks


def run_first_stage(image, net, scale, threshold):
    """ 
        Run P-Net, generate bounding boxes, and do NMS.
    """
    width, height = image.size
    sw, sh = math.ceil(width * scale), math.ceil(height * scale)
    img = image.resize((sw, sh), Image.BILINEAR)
    # img = np.asarray(img, 'float32')
    # preprocess 对图像进行归一化操作
    img = transforms.ToTensor()(img).unsqueeze(0)
    # print('img:', img)

    output = net(img)
    # 只有一张图 batch = 1，所以 [0, ,:,:]
    # [ , 1,:,:]代表 face=True 的概率
    probs = output[0].data.numpy()[0, 0, :, :]
    # offsets shape[4, o_h,o_w]
    offsets = output[1].data.numpy()
    # print('offsets:', offsets)
    # boxes
    boxes = _generate_bboxes(probs, offsets, scale, threshold)
    if len(boxes) == 0:
        return None

    # [[x1,y1,x2,y2,score,offsets],[]...]
    # 只取4个坐标加一个置信度进行nms
    keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
    return boxes[keep]


def _generate_bboxes(probs, offsets, scale, threshold):
    """
       Generate bounding boxes at places where there is probably a face.
    """
    stride = 2
    cell_size = 12

    # inds = output_feature_map [ :, :], 坐标
    inds = np.where(probs > threshold)
    '''
    >>> a =np.array([[1,2,3],[4,5,6]])
    >>> np.where(a>1)
    (array([0, 0, 1, 1, 1]), array([1, 2, 0, 1, 2]))
    '''
    # print('face candidate num'.format(len(inds)))
    if inds[0].size == 0:
        return np.array([])
    # offsets shape[4, o_h,o_w]
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    # for i in zip(tx1, ty1, tx2, ty2):
    #     print([i[j] for j in range(4)])

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]
    # print('score:', score)

    # P-Net is applied to scaled images, so we need to rescale bounding boxes back
    bounding_boxes = np.vstack([
        np.round((stride * inds[1] + 1.0) / scale),
        np.round((stride * inds[0] + 1.0) / scale),
        np.round((stride * inds[1] + 1.0 + cell_size) / scale),
        np.round((stride * inds[0] + 1.0 + cell_size) / scale),
        score, offsets
    ])
    # from
    # [[x1,x1,...]
    #  [y1,y1,...]
    #  [x2,x2,...]
    #  [y2,y2,...]
    # ]to
    # [[x1,y1,x2,y2,score,offsets],[]...]
    # shape[9,boxes_num]
    # print(bounding_boxes.shape)
    # print(bounding_boxes.T.shape)
    return bounding_boxes.T


if __name__ == '__main__':
    args = config()
    image = load_img('/Users/chenlinwei/Code/20190314mtcnn-pytorch-2/images/test3.jpg')
    bounding_boxes, landmarks = detect_faces(args,image)
    image = show_bboxes(image, bounding_boxes, landmarks)
    image.show()
