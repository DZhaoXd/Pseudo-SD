import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch.nn.functional as F
import cv2

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed
from tqdm import tqdm
import copy
from PIL import Image
import errno


# python Eval.py --config=./configs/cityscapes.yaml --labeled-id-path splits/cityscapes/train.txt --resume-path ./exp/unimatch/r101/seco_filter_64.6/

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--resume-path', type=str, default=None)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def text_save(filename, data):
    with open(filename,'w') as file:
        for i in range(len(data)):
            s = str(data[i]).replace('[','').replace(']','')
            s = s.replace("'",'').replace(',','').strip() +'\n'   
            file.write(s)
    print("{} save successful !".format(filename)) 

    
def get_color_pallete(npimg, dataset='voc'):
    out_img = Image.fromarray(npimg.astype('uint8')).convert('P')
    if dataset == 'city':
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
            255, 255, 255
        ]
        zero_pad = 256 * 3 - len(cityspallete)
        for i in range(zero_pad):
            cityspallete.append(255)
        out_img.putpalette(cityspallete)
    return out_img
    
        
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
    
def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    EVAL_HALF = False
    EVAL_HALF = True
    if EVAL_HALF:
        model = model.half()
    
    mean_prob = 0
    with torch.no_grad():
        for img, mask, id in tqdm(loader):
            img = img.cuda()

            if EVAL_HALF:
                img = img.half()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)
                prob = final.max(dim=1)[0].mean()
                mean_prob += prob
            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img).argmax(dim=1)
            
            mean_prob = mean_prob / len(loader)
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class


def pseudo_label_refine(model, loader, mode, cfg):
    EVAL_HALF = False
    EVAL_HALF = True
    if EVAL_HALF:
        model = model.half()
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']

    predicted_label = np.zeros((len(loader), 256, 512))
    predicted_prob = np.zeros((len(loader), 256, 512))
    index = 0
    image_name = []
    
    name_list= []
    vaild_dict = {}
    EXP_name = 'DTST_DIFF'
    text_save_path = './splits/cityscapes/{}/'.format(EXP_name)
    with torch.no_grad():
        for img, mask, id in tqdm(loader):
            
            img = img.cuda()
            if EVAL_HALF:
                img = img.half()
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                cnt = torch.zeros(b, 1, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        cnt[:, :, row: min(h, row + grid), col: min(w, col + grid)] += 1
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)
                    
                pred = final / cnt

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img)
            
            #### resized pred
            pred = F.interpolate(pred, size=([xx//4 for xx in pred.shape[-2:]]), mode='bilinear', align_corners=True).squeeze()
            predicted_label[index] = pred.cpu().numpy().squeeze().argmax(0).copy()
            predicted_prob[index] = np.max(pred.cpu().numpy().squeeze(), 0).copy() 
            image_name.append(id[0].split(' ')[0])
            index +=1
            
            
        thres = []
        for i in range(19):
            x = predicted_prob[predicted_label==i]
            if len(x) == 0:
                thres.append(0)
                continue        
            x = np.sort(x)
            thres.append(x[int(np.round(len(x)*0.5))])
        print(thres)
        thres = np.array(thres)
        thres[thres>0.95]=0.95
        print(thres)

        for index in range(len(image_name)):
            name = image_name[index]
            label = predicted_label[index]
            prob = predicted_prob[index]
            for i in range(19):
                label[(prob<thres[i])*(label==i)] = 255  
            output = np.asarray(label, dtype=np.uint8)
            
            mask = get_color_pallete(output, "city")
            output_folder = './dataset/Cityscapes/'
            mask_filename = name.replace('leftImg8bit/train', '{}/train'.format(EXP_name))
            save_name = os.path.join(output_folder, mask_filename)
            mkdir( os.path.dirname(save_name) )
            print('mask save to', save_name)
            mask.save(save_name)
            
            train_li = name.split(' ')[0] + ' ' + mask_filename
            name_list.append(train_li)
            vaild_dict[train_li] = np.sum(output!=255)
            
    # save training text
    ranked_name_list = sorted(name_list, key=lambda c: vaild_dict[c], reverse=True)
    topK = int(len(ranked_name_list) * 0.5)
    mkdir( os.path.dirname(text_save_path) )
    text_save(os.path.join(text_save_path, 'labeled.txt'), ranked_name_list[:topK])
    text_save(os.path.join(text_save_path, 'unlabeled.txt'), ranked_name_list[topK:])
            

name2trainid = {
    "road": 0,
    "sidewalk": 1,
    "building": 2,
    "wall": 3,
    "fence": 4,
    "pole": 5,
    "light": 6,
    "sign": 7,
    "vegetation": 8,
    "terrain": 9,
    "sky": 10 ,
    "person": 11,
    "rider": 12,
    "car": 13,
    "truck": 14,
    "bus": 15,
    "train": 16,
    "motocycle": 17,
    "bicycle": 18,
}

def diffision_refine(model, loader, mode, cfg):
    EVAL_HALF = False
    EVAL_HALF = True
    if EVAL_HALF:
        model = model.half()
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']

    predicted_label = np.zeros((len(loader), 256, 256))
    predicted_prob = np.zeros((len(loader), 256, 256))
    index = 0
    image_name = []
    
    name_list= []
    vaild_dict = {}
    EXP_name = 'Cityscapes_LIS_mask_pseudo_balance'
    text_save_path = './splits/cityscapes/{}/'.format(EXP_name)
    with torch.no_grad():
        for img, mask, id in tqdm(loader):
            
            img = img.cuda()
            if EVAL_HALF:
                img = img.half()
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                cnt = torch.zeros(b, 1, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        cnt[:, :, row: min(h, row + grid), col: min(w, col + grid)] += 1
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)
                    
                pred = final / cnt

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img)
            
            #### resized pred
            pred = F.interpolate(pred, size=([xx//2 for xx in pred.shape[-2:]]), mode='bilinear', align_corners=True).squeeze()
            predicted_label[index] = pred.cpu().numpy().squeeze().argmax(0).copy()
            predicted_prob[index] = np.max(pred.cpu().numpy().squeeze(), 0).copy() 
            image_name.append(id[0].split(' ')[0])
            index +=1
            
        thres = []
        for i in range(19):
            x = predicted_prob[predicted_label==i]
            if len(x) == 0:
                thres.append(0)
                continue        
            x = np.sort(x)
            thres.append(x[int(np.round(len(x)*0.5))])
        print(thres)
        thres = np.array(thres)
        thres[thres>0.9]=0.9
        print(thres)
        
        # Cityscapes_LIS_mask_pseudo_balance/diffison_image/aachen_000031_000019_gtFine_labelTrainIds_bus_0.png 
        # Cityscapes_LIS_mask_pseudo_balance/diffison_label/aachen_000031_000019_gtFine_labelTrainIds_bus_0.png
        for index in range(len(image_name)):
            name = image_name[index]
            label = predicted_label[index]
            prob = predicted_prob[index]
            for i in range(19):
                label[(prob<thres[i])*(label==i)] = 255  
            output = np.asarray(label, dtype=np.uint8)
            
            target_class = name2trainid[name.split('/')[-1].split('_')[-2]]
            if np.sum(output==target_class) > 500:
                mask = get_color_pallete(output, "city")
                output_folder = './dataset/Cityscapes/'
                name = name.replace('{}/'.format(EXP_name), '{}_filter/'.format(EXP_name))
                mask_filename = name.replace('diffison_image', 'diffison_label')
                save_name = os.path.join(output_folder, mask_filename)
                mkdir( os.path.dirname(save_name) )
                print('mask save to', save_name)
                mask.save(save_name)
                
                train_li = name + ' ' + mask_filename
                name_list.append(train_li)
                vaild_dict[train_li] = np.sum(output!=255)
                
    # save training text
    ranked_name_list = sorted(name_list, key=lambda c: vaild_dict[c], reverse=True)
    topK = int(len(ranked_name_list) * 0.5)
    mkdir( os.path.dirname(text_save_path) )
    text_save(os.path.join(text_save_path, 'labeled.txt'), ranked_name_list[:topK])
    text_save(os.path.join(text_save_path, 'unlabeled.txt'), ranked_name_list[topK:])
    
    


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    torch.backends.cudnn.benchmark = True
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    
    model = DeepLabV3Plus(cfg)
    model.cuda()

    trainset = SemiDataset(cfg['dataset'], cfg['data_root'], 'psd')
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    #valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'psd')
    id_path = '/data/zd/dg/FreestyleNet-main/FreestyleNet-main/splits/Cityscapes_LIS_mask_pseudo_balance/labeled.txt'
    diffision_set = SemiDataset(cfg['dataset'], cfg['data_root'], 'diff', id_path=id_path)
    
    trainloader = DataLoader(trainset, batch_size=1,
                             pin_memory=True, num_workers=1, drop_last=True, sampler=None)
                             
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=None)

    difloader = DataLoader(diffision_set, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=None)
                           
                           
    if os.path.exists(os.path.join(args.resume_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(args.resume_path, 'best.pth'))
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items(): 
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        epoch = checkpoint['epoch']
        print('************ Load from checkpoint at epoch %i\n' % epoch)
    else:
        print('ERROR FOR LOADING CKPT............')
        return

                 
    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    #mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)
    mIoU, iou_class = evaluate(model, difloader, 'original', cfg)
    #mIoU, iou_class = pseudo_label_refine(model, trainloader, eval_mode, cfg)
    #mIoU, iou_class = diffision_refine(model, difloader, 'original', cfg)
    
    for (cls_idx, iou) in enumerate(iou_class):
        logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                    'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
    logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
    
                    
if __name__ == '__main__':
    main()

