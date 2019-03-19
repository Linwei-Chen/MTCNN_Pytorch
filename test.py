from config import *
from PIL import Image
from train import config, load_net
from torchvision import transforms
from util import load_img, show_bboxes


def onet_test(args, img_path):
    img = load_img(img_path)
    net = load_net(args, 'pnet')
    output = net((transforms.ToTensor()(img.resize((12, 12), Image.BILINEAR))).unsqueeze(0))
    print('prob:', output[0])
    show_bboxes(img, [[(250 * t.item() + 250 * (i > 1)) for i, t in enumerate(output[1][0])]]).show()


def img_face_detect(args, img_path, th=[0.6, 0.7, 0.8]):
    img = None
    try:
        print('===> loading the img...')
        img = Image.open(img_path)
        img = img.convert('RGB')
    except Exception:
        print('*** warning loading fail!')
        return
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    pnet, rnet, onet = load_net(args, 'pnet'), load_net(args, 'rnet'), load_net(args, 'onet')
    resize_ratio = 0.7071
    det, box, _ = pnet(img_tensor)
    det_faces = det.ge(th[0])
    print(det)


if __name__ == '__main__':
    args = config()
    # img_face_detect(args, '/Users/chenlinwei/Desktop/屏幕快照 2019-02-24 上午8.17.14.png')
    onet_test(args, '/Users/chenlinwei/Dataset/CNN_FacePoint/train/lfw_5590/Aaron_Eckhart_0001.jpg')
