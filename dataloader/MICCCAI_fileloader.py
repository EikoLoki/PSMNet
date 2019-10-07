import torch.utils.data as data
import random
import numpy as np
import dataloader.preprocess as preprocess
import dataloader.readpfm as rp
import matplotlib.pyplot as plt
from PIL import Image


IMG_EXTENSIONS = ['.JPG', '.jpg',
                  '.PNG', '.png',
                  '.JPEG', '.jpeg']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    # print(path)
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    img = rp.readPFM(path)
    # print(path)
    # print(img[0].shape)
    # plt.imshow(img[0])
    # plt.show()
    return img


class testImageLoader(data.Dataset):
    def __init__(self, left, right, loader = default_loader):
        self.left = left
        self.right = right
        self.loader = loader

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]

        left_img = self.loader(left)
        right_img = self.loader(right)

        w, h = left_img.size

        left_up = left_img.crop((0, 0, 1280, 512))
        right_up = right_img.crop((0, 0, 1280, 512))

        left_mid = left_img.crop((0, 256, 1280, 768))
        right_mid = right_img.crop((0, 256, 1280, 768))

        left_bot = left_img.crop((0, 512, 1280, 1024))
        right_bot = right_img.crop((0, 512, 1280, 1024))

        processed = preprocess.get_transform(augment=False)
        left_up = processed(left_up)
        right_up = processed(right_up)

        left_mid = processed(left_mid)
        right_mid = processed(right_mid)

        left_bot = processed(left_bot)
        right_bot = processed(right_bot)


        return left, right, left_up, right_up, left_mid, right_mid, left_bot, right_bot

    def __len__(self):
        return len(self.left)


class myImageLoader(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader = default_loader, dploader = disparity_loader):
        self.left = left
        self.right = right
        self.disp_l = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_l = self.disp_l[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        data_l, scale_l = self.dploader(disp_l)
        data_l = np.ascontiguousarray(data_l, dtype = np.float32)

        if self.training:
            w, h = left_img.size
            th, tw = 256, 800

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            data_l = data_l[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment = False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, data_l
        else:

            left_up = left_img.crop((0, 0, 1280, 512))
            right_up = right_img.crop((0, 0, 1280, 512))

            left_mid = left_img.crop((0, 256, 1280, 768))
            right_mid = right_img.crop((0, 256, 1280, 768))

            left_bot = left_img.crop((0, 512, 1280, 1024))
            right_bot = right_img.crop((0, 512, 1280, 1024))
            #
            # left_img.show()
            # right_img.show()




            # data_l = data_l[0:384, 0:640]
            processed = preprocess.get_transform(augment=False)
            left_up = processed(left_up)
            right_up = processed(right_up)
            left_mid = processed(left_mid)
            right_mid = processed(right_mid)
            left_bot = processed(left_bot)
            right_bot = processed(right_bot)


            return left, left_up, right_up, left_mid, right_mid, left_bot, right_bot, data_l




    def __len__(self):
        return len(self.left)
