from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pickle

def transform_label_bdd100k(pred):
    Label = namedtuple('Label', [
    
        'name',  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
    
        'id',  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.
    
        'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images with train IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!
    
        'category',  # The name of the category that this label belongs to
    
        'categoryId',  # The ID of this category. Used to create ground truth images
        # on category level.
    
        'hasInstances',  # Whether this label distinguishes between single instances or not
    
        'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not
    
        'color',  # The color of this label
    ])
            
    labels = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 154)),  # (153,153,153)
        Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 143)),  # (  0,  0,142)
    ]
    
    bdd_trainId2trainId = {label.trainId: label.trainId for label in labels}
    
    pred_copy = 255 * np.ones(pred.shape, dtype=np.float32)
    for k, v in bdd_trainId2trainId.items():
        pred_copy[pred == k] = v    
    return pred_copy

def transform_label(pred):
    synthia_to_city = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            10: 9,
            11: 10,
            12: 11,
            13: 12,
            15: 13,
            17: 14,
            18: 15,
        }
    label_copy = 255 * np.ones(pred.shape, dtype=np.float32)
    for k, v in synthia_to_city.items():
        label_copy[pred == k] = v
    return label_copy.copy()


class rand_mixer():
    def __init__(self,  target_name, root='./dataset/Cityscapes/', dataset='cityscapes',class_num=19):
        
        self.root = root
        self.class_num = class_num
        self.target_name = target_name
        self.target_name = 'seco_filter'
        self.target_name = 'Cityscapes_LIS_mask_pseudo_balance_filter'
        self.target_name = 'Cityscapes_LIS_mask_pseudo_over_x5'
        if dataset == "cityscapes":
            self.file_list = os.path.join(root, self.target_name, '{}.p'.format(self.target_name))
            #self.base_label_path = self.root
            #self.base_img_path = os.path.join(self.root, "leftImg8bit/train/")
            self.base_label_path = self.root
            self.base_img_path = self.root
        else:
            print('rand_mixer {} unsupported'.format(dataset))
            return
            
        self.label_to_file, _ = pickle.load(open(self.file_list, "rb"))
        
    def oneMix(self, mask, data = None, target = None):
        #Mix
        if not (data is None):
            data = (mask*data[0]+(1-mask)*data[1]).unsqueeze(0)
        if not (target is None):
            target = (mask*target[0]+(1-mask)*target[1]).unsqueeze(0)
        return data, target
    
    def generate_class_mask(self, pred, classes):
        pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
        N = pred.eq(classes).sum(0)
        return N

    def mix(self, in_imgs, in_lbls, classes, choice_p, out_size, ignore_value, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
        out_imgs = torch.ones_like(in_imgs)
        out_lbls = torch.ones_like(in_lbls)
        bs_idx = 0
        for (in_img, in_lbl) in zip(in_imgs, in_lbls):
            #class_idx = random.sample(classes, 1)[0]
            class_idx = np.random.choice(classes, size=1, p=choice_p)[0]
            while True:
                name = random.sample(self.label_to_file[class_idx], 1)
                
                #img_path = os.path.join(self.base_img_path, name[0].split('/')[-2], name[0].split('/')[-1])
                #label_path = os.path.join(self.base_label_path, name[0])

                img_path = os.path.join(self.base_img_path, name[0].strip().split(' ')[0])
                label_path = os.path.join(self.base_label_path, name[0].strip().split(' ')[1])

                img = Image.open(img_path).convert('RGB')
                lbl = Image.open(label_path)

                
                count=10
                while count>0:
                    #img_, mask_ = resize(img, lbl, (0.5, 2.0))
                    img_, mask_ = resize(img, lbl, (0.75, 1.25))
                    img_, mask_ = crop(img_, mask_, out_size, ignore_value)
            
                    img_ = np.asarray(img_, np.float32)
                    mask_ = np.asarray(mask_, np.float32)
                    #print("selected class, ", class_idx)
                    #print("mask_ unique, ", np.unique(mask_))
                    if class_idx in mask_:
                        break
                    count = count-1
                img = img_.copy()
                lbl = mask_.copy()
                break
                
            img = img.copy() / 255 
            img -= PIXEL_MEAN
            img /= PIXEL_STD
            img = img.transpose((2, 0, 1))
            img = torch.Tensor(img)
            lbl = torch.Tensor(lbl)

            class_i = torch.Tensor([class_idx]).type(torch.int64)
            MixMask = self.generate_class_mask(lbl, class_i)
           
            if self.class_num==19:
                if class_idx == 12:
                    if 17 in lbl:
                        MixMask += self.generate_class_mask(lbl, torch.Tensor([17]).type(torch.int64))
                    if 18 in lbl:
                        MixMask += self.generate_class_mask(lbl, torch.Tensor([18]).type(torch.int64))

                if class_idx == 17 or class_idx == 18:
                    MixMask += self.generate_class_mask(lbl, torch.Tensor([12]).type(torch.int64))

                if class_idx == 6 or class_idx == 7:
                    if 5 in lbl:
                        MixMask += self.generate_class_mask(lbl, torch.Tensor([5]).type(torch.int64))

            else:
                if class_idx == 11:
                    if 14 in lbl:
                        MixMask += self.generate_class_mask(lbl, torch.Tensor([14]).type(torch.int64))
                    if 15 in lbl:
                        MixMask += self.generate_class_mask(lbl, torch.Tensor([15]).type(torch.int64))

                if class_idx == 14 or class_idx == 15:
                    MixMask += self.generate_class_mask(lbl, torch.Tensor([11]).type(torch.int64))
            mixdata = torch.cat((img.unsqueeze(0), in_img.unsqueeze(0)))
            mixtarget = torch.cat((lbl.unsqueeze(0), in_lbl.unsqueeze(0)))
            data, target = self.oneMix(MixMask, data=mixdata, target=mixtarget)
            out_imgs[bs_idx] = data
            out_lbls[bs_idx] = target
            bs_idx += 1
        return out_imgs, out_lbls


def denormalizeimage(images, mean=(0., 0., 0.), std=(1., 1., 1.)):
    """Denormalize tensor images with mean and standard deviation.
    Args:
        images (tensor): N*C*H*W
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    images = images.cpu().numpy()
    # N*C*H*W to N*H*W*C
    images = images.transpose((0,2,3,1))
    images *= std
    images += mean
    images *=255.0
    # N*H*W*C to N*C*H*W
    images = images.transpose((0,3,1,2))
    return torch.tensor(images)
    
class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            if mode == 'train_l':
                # class 19
                self.mixer = rand_mixer(target_name=self.name, class_num=19)
                self.mix_classes = [4, 5, 6, 7, 12, 14, 15, 16, 17, 18]
                
                # class 16
                if self.name == 'cityscapes_16':
                    self.mixer = rand_mixer(target_name=id_path.split('/')[-2], class_num=16)
                    self.mix_classes = [3, 4, 5, 6, 7, 12, 14, 15]

                """
                image_freq = [1478, 789, 1453, 298, 156, 671, 317, 152, 1402, 358, 1253, 705, 109, 1388, 211, 74, 13, 52, 122]
                image_freq = [1193, 871, 1409, 392, 221, 955, 407, 157, 1398, 496, 1313, 793, 129, 1401, 291, 85, 26, 57, 143]

                self.mix_classes = [3, 4, 6, 7, 9, 12, 14, 15, 16, 17, 18]
                self.class_image_freq = []
                for x in self.mix_classes:
                    self.class_image_freq.append(image_freq[x])
                self.mix_p = 1 / (np.array(self.class_image_freq) / np.array(self.class_image_freq).sum())
                self.mix_p = np.exp(1 - np.array(self.class_image_freq) / np.array(self.class_image_freq).sum() )
                self.mix_p = np.array(self.mix_p) / np.array(self.mix_p).sum()
                """
                self.mix_p = np.ones(len(self.mix_classes)) / len(self.mix_classes)
                print(self.mix_p)

            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        elif mode == 'val':
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()
        elif mode == 'psd':
            with open('splits/%s/train.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()
        else:
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
                
    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        if self.mode == 'val' or self.mode == 'psd' or self.mode == 'diff':
            if self.name == 'cityscapes_16':
                mask = Image.fromarray(transform_label(np.array(mask)))
            if self.name == 'bdd':
                mask = Image.fromarray(transform_label_bdd100k(np.array(mask)))
            img, mask = normalize(img, mask)
            return img, mask, id

        if img.size[0] < self.size or img.size[1] < self.size:
            img, mask = resize(img, mask, (0.75, 1.25))
        else:
            img, mask = resize(img, mask, (0.5, 2.0))
            
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            img, mask = normalize(img, mask)
            
            ## show
            """
            PIXEL_MEAN=[0.485, 0.456, 0.406]
            PIXEL_STD=[0.229, 0.224, 0.225]
            denormalized_image = denormalizeimage(img.clone().unsqueeze(0), PIXEL_MEAN, PIXEL_STD)
            denormalized_image = np.asarray(denormalized_image.cpu().numpy()[0], dtype=np.uint8)
            denormalized_image = Image.fromarray(denormalized_image.transpose((1,2,0)))
            denormalized_image.save('./show_res/o_{}'.format(id.split(' ')[0].split('/')[-1] ))
            """
            if random.random() < 0.9:
                img, mask = self.mixer.mix(img.unsqueeze(0), mask.unsqueeze(0), self.mix_classes, self.mix_p, self.size, ignore_value)
                img, mask = img.squeeze(0), mask.squeeze(0)
            """
            denormalized_image = denormalizeimage(img.clone().unsqueeze(0), PIXEL_MEAN, PIXEL_STD)
            denormalized_image = np.asarray(denormalized_image.cpu().numpy()[0], dtype=np.uint8)
            denormalized_image = Image.fromarray(denormalized_image.transpose((1,2,0)))
            denormalized_image.save('./show_res/m_{}'.format(id.split(' ')[0].split('/')[-1] ))
            print(img.shape)
            """
            return img, mask

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
