import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
import PIL
import random
from PIL import Image, ImageOps, ImageFilter
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.COCO import COCO_dict
from ldm.data.ADE20K import ADE20K_dict
from ldm.data.Cityscapes import Cityscapes_dict

from torch.utils.data import DataLoader, Dataset
import pickle


def get_color_pallete(npimg, dataset='Cityscapes'):
    out_img = Image.fromarray(npimg.astype('uint8')).convert('P')
    if dataset == 'Cityscapes':
        cityspallete = [
            128, 64, 128,
            244, 35, 232,
            70, 70, 70,
            102, 102, 156,
            190, 153, 153,
            153, 153, 153,
            250, 170, 30,
            220, 220, 0,
            107, 142, 35,
            152, 251, 152,
            0, 130, 180,
            220, 20, 60,
            255, 0, 0,
            0, 0, 142,
            0, 0, 70,
            0, 60, 100,
            0, 80, 100,
            0, 0, 230,
            119, 11, 32,
        ]
        zero_pad = 256 * 3 - len(cityspallete)
        for i in range(zero_pad):
            cityspallete.append(255)
        out_img.putpalette(cityspallete)
    if dataset == 'ADE20K':
        PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
               [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
               [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
               [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
               [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
               [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
               [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
               [102, 255, 0], [92, 0, 255]]
        PALETTE_ = [item for sublist in PALETTE for item in sublist]
        zero_pad = 256 * 3 - len(PALETTE_)
        for i in range(zero_pad):
            PALETTE_.append(255)
        out_img.putpalette(PALETTE_)        
    return out_img

    
class COCOVal(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=512,
                 interpolation="bicubic",
                 ):
        self.data_root = data_root
        self.data_paths = txt_file
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        path_ = self.image_paths[i][:-4]
        path2 = os.path.join(self.data_root, 'stuffthingmaps_trainval2017/val2017', path_.split('/')[-1] + '.png')
        pil_image2 = Image.open(path2)
        example["img_name"] = path_.split('/')[-1]

        pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        label = np.array(pil_image2).astype(np.float32)
        example["label"] = label
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        if class_ids[-1] == 255:
            class_ids = class_ids[:-1]
        class_ids_final = np.zeros(182)
        text = ''
        for i in range(len(class_ids)):
            text += COCO_dict[str(class_ids[i])]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]
        example["caption"] = text
        example["class_ids"] = class_ids_final

        return example

class ADE20KVal(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=512,
                 interpolation="bicubic",
                 ):
        self.data_root = data_root
        self.data_paths = txt_file
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        path_ = self.image_paths[i][:-4]
        path2 = os.path.join(self.data_root, path_.split('.')[-1] + '.png')
        pil_image2 = Image.open(path2)
        example["img_name"] = path_.split('/')[-1]

        pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        label = np.array(pil_image2).astype(np.float32)
        label += 1
        example["label"] = label
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        print(class_ids)
        if class_ids[0] == 0:
            class_ids = class_ids[1:]
        class_ids_final = np.zeros(151)
        text = ''
        for i in range(len(class_ids)):
            text += ADE20K_dict[str(class_ids[i])]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]
        example["caption"] = text
        example["class_ids"] = class_ids_final

        return example


class CityscapesVal(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=512,
                 interpolation="bicubic",
                 crop=True
                 ):
        self.data_root = data_root
        self.data_paths = txt_file
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.size = size
        self.crop = crop
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    def get_params(self, img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw
        
    def __getitem__(self, i):
        example = dict()
        path = os.path.join(self.data_root, 'leftImg8bit', 'train', self.image_paths[i])
        path2 = os.path.join(self.data_root, 'gtFine', 'train', self.image_paths[i].replace('leftImg8bit.png', 'gtFine_labelTrainIds.png')) # add replace
        pil_image2 = Image.open(path2)
        path_ = self.image_paths[i]
        example["img_name"] = path_.split('/')[-1]

        
        if self.crop:
            w, h = pil_image2.size
            w, h = w//2, h//2
            pil_image2 = pil_image2.resize((w, h), resample=PIL.Image.NEAREST)
            i, j, th, tw = self.get_params(pil_image2, (self.size, self.size))
            pil_image2 = pil_image2.crop((j,i,tw+j, th+i))            
        else:
            pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)

            
        label = np.array(pil_image2).astype(np.float32)
        example["label"] = label
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        class_ids_final = np.zeros(len(Cityscapes_dict))
        text = ''
        #domain_caption = 'a sketch of '
        #domain_caption = 'an ink painting of'
        #text += domain_caption
        for i in range(len(class_ids)):
            if class_ids[i] == 255:
                continue
            text += Cityscapes_dict[class_ids[i]]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        
        text = text[:-1]
        example["caption"] = text
        example["class_ids"] = class_ids_final

        return example


def crop(img, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))

    return img

def resize(mask, ratio_range):
    w, h = mask.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    mask = mask.resize((ow, oh), Image.NEAREST)
    return mask
    
class CityscapesBalance(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=512,
                 interpolation="bicubic",
                 crop=True,
                 max_iters=None,
                 num_class=19
                 ):
        self.data_root = data_root
        self.size = size
        self.crop = crop
        self.max_iters = max_iters
        self.num_class = num_class
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        if self.max_iters is not None:
            ### txt_file is a frenquency .p file
             self.label_to_file, self.file_to_label = pickle.load(open(txt_file, "rb"))
             self.image_paths = []
             SUB_EPOCH_SIZE = 500
             tmp_list = []
             ind = dict()
             
             selected_class_list = [3, 4, 12, 14, 15, 16, 17, 18]
             selected_class_dict = {}
             for idx, val in enumerate(selected_class_list):
                 selected_class_dict[val] = idx
                 
             for i in selected_class_list:
                 ind[i] = 0
             for e in range(int(self.max_iters / SUB_EPOCH_SIZE) + 1):
                 cur_class_dist = np.zeros(len(selected_class_list))
                 for i in range(SUB_EPOCH_SIZE):
                     if cur_class_dist.sum() == 0:
                         dist1 = cur_class_dist.copy()
                     else:
                         dist1 = cur_class_dist / cur_class_dist.sum()
                     w = 1 / np.log(1 + 1e-2 + dist1)
                     w = w / w.sum()
                     c = np.random.choice(selected_class_list, p=w)
        
                     if len(self.label_to_file[c]) == 0 or len(self.label_to_file[c]) == 1:
                         continue

                     if ind[c] > (len(self.label_to_file[c]) - 1):
                         np.random.shuffle(self.label_to_file[c])
                         ind[c] = ind[c] % (len(self.label_to_file[c]) - 1)
        
                     c_file = self.label_to_file[c][ind[c]]
                     
                     tmp_list.append(c_file + ' ' + str(c))
                     ind[c] = ind[c] + 1
                     
                     for x in self.file_to_label[c_file]:
                        if x in selected_class_dict:
                            cur_class_dist[selected_class_dict[x]] += 1
             print("------------------------city balance sample-----------------------------")
             self.image_paths = tmp_list
        else:
            ### txt_file is a frenquency .p file
            with open(txt_file, "r") as f:
                self.image_paths = f.read().splitlines()
             
        self._length = len(self.image_paths)

    def __len__(self):
        return self._length
    
    
    def __getitem__(self, i):
        example = dict()
        path_, target_c = self.image_paths[i].split(' ')
        target_name=Cityscapes_dict[int(target_c)]
        path2 = os.path.join(self.data_root, path_)
        pil_image2 = Image.open(path2)
        example["img_name"] = path_.split('/')[-1].replace('.png', '_'+target_name+'.png')
        # resize and crop
        if self.crop:
            pil_image2 = resize(pil_image2, (0.5, 0.75))
            cnt=10000
            while cnt > 0:
                pil_image_temp = crop(pil_image2, self.size)
                if np.sum(np.array(pil_image_temp)==target_c)>250:
                    pil_image2 = pil_image_temp
                    break
                cnt -= 1
            if cnt == 0:
                pil_image2 = crop(pil_image2, self.size)

        else:
            pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)

        label = np.array(pil_image2).astype(np.float32)
        example["label"] = label
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        class_ids_final = np.zeros(len(Cityscapes_dict))
        text = ''
        for i in range(len(class_ids)):
            if class_ids[i] == 255:
                continue
            text += Cityscapes_dict[class_ids[i]]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        
        text = text[:-1]
        example["caption"] = text
        example["class_ids"] = class_ids_final

        return example

class CityscapesOverSample(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=512,
                 interpolation="bicubic",
                 crop=True,
                 max_iters=None,
                 num_class=19
                 ):
        self.data_root = data_root
        self.size = size
        self.crop = crop
        self.max_iters = max_iters
        self.num_class = num_class
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        if self.max_iters is not None:
            ### txt_file is a frenquency .p file
             self.label_to_file, self.file_to_label = pickle.load(open(txt_file, "rb"))
             image_freq = [2934, 2737, 2925, 824, 1144, 2802, 1058, 2291, 2869, 1321, 2617, 1839, 604, 2754, 266, 220, 124, 296, 1153]
             mix_p = 1 / (np.array(image_freq) / np.array(image_freq).sum())
             mix_p = np.exp( (1 - np.array(image_freq) / np.array(image_freq).sum()) * len(image_freq) * 5) 
             mix_p = np.array(mix_p) / np.array(mix_p).sum()
             self.image_paths = []
             for iter_num in range(self.max_iters):
                c = np.random.choice(list(range(len(image_freq))), p=mix_p)
                c_file = np.random.choice(self.label_to_file[c])
                self.image_paths.append(c_file + ' ' + str(c))
        else:
            ### txt_file is a frenquency .p file
            with open(txt_file, "r") as f:
                self.image_paths = f.read().splitlines()
             
        self._length = len(self.image_paths)

    def __len__(self):
        return self._length
    
    
    def __getitem__(self, i):
        example = dict()
        
        while True:
            path_, target_c = self.image_paths[i].split(' ')
            print(path_)
            target_c = int(target_c)
            target_name=Cityscapes_dict[int(target_c)]
            path2 = os.path.join(self.data_root, path_)
            pil_image2 = Image.open(path2)
            #print('self.image_paths[i]',self.image_paths[i])
            #print('+', np.sum(np.array(pil_image2)==target_c))
            pil_image2 = pil_image2.resize((2048, 1024))
            pil_image2 = resize(pil_image2, (0.5, 0.75))
            #print('-', np.sum(np.array(pil_image2)==target_c))
            #print(np.unique(np.array(pil_image2)))
            if self.crop:
                cnt=100
                while cnt > 0:
                    pil_image_temp = crop(pil_image2, self.size)
                    if np.sum(np.array(pil_image_temp)==target_c)>250:
                        pil_image2 = pil_image_temp
                        break
                    cnt -= 1
                if cnt == 0:
                    pil_image2 = crop(pil_image2, self.size)
            else:
                pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            if np.sum(np.array(pil_image_temp)==target_c)>50:
                break
            else:
                i = np.random.randint(0, len(self.image_paths)-1)
        
            
        example["img_name"] = path_.split('/')[-1].replace('.png', '_'+target_name+'.png')
        label = np.array(pil_image2).astype(np.float32)
        example["label"] = label
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        class_ids_final = np.zeros(len(Cityscapes_dict))
        text = ''
        for i in range(len(class_ids)):
            if class_ids[i] == 255:
                continue
            text += Cityscapes_dict[class_ids[i]]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        
        text = text[:-1]
        example["caption"] = text
        example["class_ids"] = class_ids_final

        return example
        
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/layout2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size",
    )
    parser.add_argument(
        "--out_num",
        type=int,
        default=500,
        help="out numbers",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-finetune_COCO.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        required=True, 
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--txt_file",
        type=str,
        required=True,
        help="path to txt file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="which dataset to evaluate",
        choices=["COCO", "ADE20K", 'Cityscapes', 'CityscapesBalance', 'CityscapesOver'],
        default="COCO"
    )

    
    opt = parser.parse_args()

    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.batch_size
    if opt.dataset == "COCO":
        val_dataset = COCOVal(data_root=opt.data_root, txt_file=opt.txt_file)
    elif opt.dataset == "ADE20K":
        val_dataset = ADE20KVal(data_root=opt.data_root, txt_file=opt.txt_file)
    elif opt.dataset == "Cityscapes":
        val_dataset = CityscapesVal(data_root=opt.data_root, txt_file=opt.txt_file)
    elif opt.dataset == "CityscapesBalance":
        val_dataset = CityscapesBalance(data_root=opt.data_root, txt_file=opt.txt_file, max_iters=opt.out_num*opt.batch_size,
                         num_class=19)
    elif opt.dataset == "CityscapesOver":
        val_dataset = CityscapesOverSample(data_root=opt.data_root, txt_file=opt.txt_file, max_iters=opt.out_num*opt.batch_size,
                         num_class=19)
        
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    name_dict = {}
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for data in val_dataloader:
                    label = data["label"].to(device)
                    class_ids = data["class_ids"].to(device)
                    text = data["caption"]
                    print(text)
                    c = model.get_learned_conditioning(text)
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     label=label,
                                                     class_ids=class_ids,
                                                     batch_size=batch_size,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    

                    for i in range(len(x_samples_ddim)):
                        x_sample = x_samples_ddim[i]
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img_name = data["img_name"][i]
                        
                        if img_name not in name_dict:
                            name_dict[img_name]=0
                        else:
                            name_dict[img_name]=name_dict[img_name]+1
                        
                        #x_sample = cv2.resize(x_sample, None, fx=2.0, fy=1.0, interpolation = cv2.INTER_NEAREST)
                        save_name = "{}_{}.png".format(img_name.replace('.png',''), str(name_dict[img_name]))
                        os.makedirs(os.path.join(outpath, 'diffison_image'), exist_ok=True)
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(outpath, 'diffison_image', f"{save_name}") )
                        
                        label_show = get_color_pallete(label[i].cpu().numpy(), opt.dataset)
                        os.makedirs(os.path.join(outpath, 'diffison_label'), exist_ok=True)
                        label_show.save(
                            os.path.join(outpath, 'diffison_label', f"{save_name}"))
                            
                        combine_show=True
                        #combine_show=False
                        if combine_show:
                            print(label.cpu().numpy().shape)
                            label_show = get_color_pallete(label[i].cpu().numpy(), opt.dataset)
                            x_sample_show = Image.fromarray(x_sample.astype(np.uint8))
                            show_list = [x_sample_show, label_show]
                            h, w = show_list[0].size
                            show_img = Image.new("RGB", (h*len(show_list) + len(show_list)*20, w), 'white')
                            for i, im in enumerate(show_list):
                                im = im.resize((h, w), Image.NEAREST)
                                show_img.paste(im, box=( i * h + 20 * i, 0))
                            os.makedirs(os.path.join(outpath, 'diffison_show'), exist_ok=True)
                            show_img.save(os.path.join(outpath, 'diffison_show',  f"{save_name}"))                            
                        
                            
                            
                            
if __name__ == "__main__":
    main()
