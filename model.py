from config import *


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


class P_Net(nn.Module):
    def __init__(self):
        super(P_Net, self).__init__()
        self.pre_layer = nn.Sequential(
            # 12x12x3
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # PReLU1
            # 10x10x10
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool1
            # 5x5x10
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # conv2
            # 3x3x16
            nn.PReLU(),  # PReLU2
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv3
            # 1x1x32
            nn.PReLU()  # PReLU3
        )
        # detection
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        # bounding box regresion
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        # landmark localization
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)
        # weight initiation with xavier
        self.apply(weights_init)

    def forward(self, x):
        x = self.pre_layer(x)
        det = torch.sigmoid(self.conv4_1(x))
        box = self.conv4_2(x)
        landmark = self.conv4_3(x)
        # det:[,2,1,1], box:[,4,1,1], landmark:[,10,1,1]
        return det, box, landmark


class R_Net(nn.Module):
    def __init__(self):
        super(R_Net, self).__init__()
        self.pre_layer = nn.Sequential(
            # 24x24x3
            nn.Conv2d(3, 28, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            # 22x22x28
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            # 10x10x28
            nn.Conv2d(28, 48, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            # 8x8x48
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            # 3x3x48
            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # conv3
            # 2x2x64
            nn.PReLU()  # prelu3
        )
        # 2x2x64
        self.conv4 = nn.Linear(64 * 2 * 2, 128)  # conv4
        # 128
        self.prelu4 = nn.PReLU()  # prelu4
        # detection
        self.conv5_1 = nn.Linear(128, 1)
        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)
        # lanbmark localization
        self.conv5_3 = nn.Linear(128, 10)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)
        det = torch.sigmoid(self.conv5_1(x))
        box = self.conv5_2(x)
        landmark = self.conv5_3(x)
        return det, box, landmark


class O_Net(nn.Module):
    def __init__(self):
        super(O_Net, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
            nn.PReLU(),  # prelu3
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool3
            nn.Conv2d(64, 128, kernel_size=2, stride=1),  # conv4
            nn.PReLU()  # prelu4
        )
        self.conv5 = nn.Linear(128 * 2 * 2, 256)  # conv5
        self.prelu5 = nn.PReLU()  # prelu5
        # detection
        self.conv6_1 = nn.Linear(256, 1)
        # bounding box regression
        self.conv6_2 = nn.Linear(256, 4)
        # lanbmark localization
        self.conv6_3 = nn.Linear(256, 10)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        # detection
        det = torch.sigmoid(self.conv6_1(x))
        box = self.conv6_2(x)
        landmark = self.conv6_3(x)
        return det, box, landmark


class LossFn:
    def __init__(self, cls_factor=1, box_factor=1, landmark_factor=1):
        # loss function
        self.cls_factor = cls_factor
        self.box_factor = box_factor
        self.land_factor = landmark_factor
        # binary cross entropy:
        self.loss_cls = nn.BCELoss()
        # mean square error
        # i.e. ||x-y||^2_2
        self.loss_box = nn.MSELoss()
        self.loss_landmark = nn.MSELoss()

    # gt=ground_truth
    def cls_loss(self, gt_label, pred_label):
        # pred_label: [batch_size, 1, 1, 1] to [batch_size]
        pred_label = torch.squeeze(pred_label)
        # gt_label: [batch_size, 1] to [batch_size ]
        gt_label = torch.squeeze(gt_label)
        # get the mask element which >= 0, only 0 and 1 can effect the detection loss
        # mask = torch.ge(gt_label, 0)
        mask = torch.ge(gt_label, 0)
        # valid_gt_label = torch.masked_select(gt_label, mask)
        # valid_pred_label = torch.masked_select(pred_label, mask)
        valid_gt_label = gt_label[mask]
        valid_pred_label = pred_label[mask]
        # print(valid_pred_label, valid_gt_label)
        return self.loss_cls(valid_pred_label, valid_gt_label) * self.cls_factor
        # return torch.tensor([0.])

    def box_loss(self, gt_label, gt_offset, pred_offset):
        # if gt_label is torch.tensor([0.0]):
        #     return torch.tensor([0.0])
        # pred_offset: [batch_size, 4] to [batch_size,4]
        pred_offset = torch.squeeze(pred_offset)
        # gt_offset: [batch_size, 4, 1, 1] to [batch_size,4]
        gt_offset = torch.squeeze(gt_offset)
        # gt_label: [batch_size, 1, 1, 1] to [batch_size]
        gt_label = torch.squeeze(gt_label)

        # get the mask element which != 0
        # unmask = torch.eq(gt_label, 0)
        # mask = torch.eq(unmask, 0)
        mask = torch.eq(gt_label, 1)
        # convert mask to dim index
        '''
        >> > torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
                                         [0.0, 0.4, 0.0, 0.0],
                                         [0.0, 0.0, 1.2, 0.0],
                                         [0.0, 0.0, 0.0, -0.4]]))
        tensor([[0, 0],
                [1, 1],
                [2, 2],
                [3, 3]])
        '''
        # chose_index = torch.nonzero(mask.data)
        # chose_index = torch.squeeze(chose_index)
        # only valid element can effect the loss
        # print('chose_index', chose_index)
        # valid_gt_offset = gt_offset[chose_index, :]
        # valid_pred_offset = pred_offset[chose_index, :]
        valid_gt_offset = gt_offset[mask, :]
        valid_pred_offset = pred_offset[mask, :]
        # print('valid_gt_offset', valid_gt_offset, 'valid_pred_offset', valid_pred_offset)
        valid_sample_num = valid_gt_offset.shape[0]
        if 0 == valid_sample_num:
            # print('No box')
            # return self.loss_box(torch.tensor([0.0]), torch.tensor([0.0]))
            return torch.tensor([0.0])
        else:
            # print('valid_sample_num', valid_sample_num)
            return self.loss_box(valid_pred_offset, valid_gt_offset) * self.box_factor
        # return torch.tensor([0.])

    def landmark_loss(self, landmark_flag, gt_landmark=None, pred_landmark=None):
        # pred_landmark:[batch_size,10,1,1] to [batch_size,10]
        pred_landmark = torch.squeeze(pred_landmark)
        # gt_landmark:[batch_size,10] to [batch_size,10]
        gt_landmark = torch.squeeze(gt_landmark)
        # gt_label:[batch_size,1] to [batch_size]
        gt_label = torch.squeeze(landmark_flag)
        mask = torch.eq(gt_label, 1)
        # chose_index = torch.nonzero(mask.data)
        # chose_index = torch.squeeze(chose_index)
        # valid_gt_landmark = gt_landmark[chose_index, :]
        # valid_pred_landmark = pred_landmark[chose_index, :]
        valid_gt_landmark = gt_landmark[mask, :]
        valid_pred_landmark = pred_landmark[mask, :]
        # print('valid_pred_landmark', valid_pred_landmark, 'valid_gt_landmark', valid_gt_landmark)
        valid_sample_num = valid_gt_landmark.shape[0]
        if 0 == valid_sample_num:
            return torch.tensor([0.0])
        else:
            return self.loss_landmark(valid_pred_landmark, valid_gt_landmark) * self.land_factor

    def total_loss(self, gt_label, pred_label, gt_offset, pred_offset, landmark_flag, gt_landmark, pred_landmark):
        return self.cls_loss(gt_label, pred_label) \
               + self.box_loss(gt_label, gt_offset, pred_offset) \
               + self.landmark_loss(landmark_flag, gt_landmark, pred_landmark)


if __name__ == '__main__':
    print(torch.tensor([0.0]).type())
