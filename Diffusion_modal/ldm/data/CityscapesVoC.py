import os
from re import L
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random
import json

def read_json_file_standard(file_path):
    try:
        with open(file_path, 'r') as file:
            # 读取 JSON 数据
            data = json.load(file)
            print("JSON 数据加载成功！", file_path)
            return data
    except FileNotFoundError:
        print("文件未找到：", file_path)
        return None
    except json.JSONDecodeError as e:
        print("JSON 解析错误：", e)
        return None
    
    
# Some words may differ from the class names defined in COCO-Stuff to minimize ambiguity
Cityscapes_dict = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motocycle",
            18: "bicycle",
            19: "boat",
            20: "bottle",
            21: "cat",
            22: "chair",
            23: "cow",
            24: "dog",
            25: "sheep",             
        }

class CityscapesVoCBase(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_root = ''
        self.data_paths = txt_file
        data_json = read_json_file_standard(self.data_paths)
        self.image_paths, self.mask_paths = [], []


        for data in data_json:
            self.image_paths.append(data['file_name'])
            self.mask_paths.append(data['layout_name'])

        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip_p = flip_p
        self.train_or_val = 'train' if 'train' in txt_file else 'val'
        
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
        
        #### 读图
        path = os.path.join(self.data_root, self.image_paths[i])
        pil_image = Image.open(path)
        if not pil_image.mode == "RGB":
            pil_image = pil_image.convert("RGB")

        pil_image2 = Image.open(os.path.join(self.data_root, self.mask_paths[i]))
        image_name = self.mask_paths[i].split('/')[-1]
        
        if image_name.startswith('20'):
            pil_image = pil_image.resize((self.size, self.size), resample=self.interpolation)
            pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        else:
            if self.size is not None:
                w, h = pil_image.size
                w, h = w//2, h//2
                pil_image = pil_image.resize((w, h), resample=self.interpolation)
                pil_image2 = pil_image2.resize((w, h), resample=PIL.Image.NEAREST)
                i, j, th, tw = self.get_params(pil_image, (self.size, self.size))
                #pil_image = pil_image.crop((i, j, th+i, tw+j))
                #pil_image2 = pil_image2.crop((i, j, th+i, tw+j))
                pil_image = pil_image.crop((j,i,tw+j, th+i))
                pil_image2 = pil_image2.crop((j,i,tw+j, th+i))
            
        flip = random.random() < self.flip_p
        if flip:
            pil_image = pil_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            pil_image2 = pil_image2.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        image = np.array(pil_image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        label = np.array(pil_image2).astype(np.float32)
        if image_name.startswith('20'):
            label[label!=255] += 19
        example["label"] = label
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        if class_ids[-1] == 255:
            class_ids = class_ids[:-1]
        class_ids_final = np.zeros(len(Cityscapes_dict))
        text = ''
        for i in range(len(class_ids)):
            text += Cityscapes_dict[class_ids[i]] # ori code: text += Cityscapes_dict[str(class_ids[i])]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]
        example["caption"] = text
        example["class_ids"] = class_ids_final

        return example


class CityscapesVoCTrain(CityscapesVoCBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CityscapesVoCValidation(CityscapesVoCBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(flip_p=flip_p, **kwargs)
