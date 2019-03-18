from config import *
from PIL import Image
from train import config, load_net
from torchvision import transforms


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
    img_face_detect(args, '/Users/chenlinwei/Desktop/屏幕快照 2019-02-24 上午8.17.14.png')
