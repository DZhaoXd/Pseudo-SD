# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support debug_output_attention

import os.path as osp
import os
import tempfile

import torch.nn.functional as F

from PIL import Image
import mmcv
import numpy as np
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

def text_save(filename, data):
    with open(filename,'w') as file:
        for i in range(len(data)):
            s = str(data[i]).replace('[','').replace(']','')
            s = s.replace("'",'').replace(',','') +'\n'   
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
        ]
#         palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
#            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
#            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
#         zero_pad = 256 * 3 - len(palette)
#         for i in range(zero_pad):
#             palette.append(0)
        out_img.putpalette(cityspallete)
    else:
        vocpallete = _getvocpallete(256)
        out_img.putpalette(vocpallete)
    return out_img
    
def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    if efficient_test:
        mmcv.mkdir_or_exist('.efficient_test')
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                if hasattr(model.module.decode_head,
                           'debug_output_attention') and \
                        model.module.decode_head.debug_output_attention:
                    # Attention debug output
                    mmcv.imwrite(result[0] * 255, out_file)
                else:
                    model.module.show_result(
                        img_show,
                        result,
                        palette=dataset.PALETTE,
                        show=show,
                        out_file=out_file,
                        opacity=opacity)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result, tmpdir='.efficient_test')
            results.append(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def diffision_single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    if efficient_test:
        mmcv.mkdir_or_exist('.efficient_test')
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
            
            label_path = data[0]['filename'].replace('diffison_image', 'diffison_label')
            label = np.array(Image.open(label_path))
            print('xxxxxxxxxxxxxxxxxxxxxxxxxx', data)
            print('result.shape', result.shape)
            print('label.shape', label.shape)

            
            from skimage.measure import label as sklabel
            labelnp = label.astype(int) 
            seg, forenum = sklabel(labelnp, background=0, return_num=True, connectivity=2)
            for i in range(forenum):
                instance_id = i+1
                ins_mask = seg==instance_id
                
                Proportion = []
                pred_unique = np.unique(result[(ins_mask>0)])
                for cid in pred_unique:
                    Proportion.append(np.mean(result[(ins_mask>0)] == cid))   
                pred_max_idx, pred_max_pro = np.argmax(np.array(Proportion)), np.max(np.array(Proportion))
                
                Proportion = []
                label_unique = np.unique(label[(ins_mask>0)])
                for cid in label_unique:
                    Proportion.append(np.mean(label[(ins_mask>0)] == cid))   
                label_max_idx, label_max_pro = np.argmax(np.array(Proportion)), np.max(np.array(Proportion))

                if pred_unique[pred_max_idx] != label_unique[label_max_idx]:
                    label[ins_mask>0] = 255
                # if label_unique[label_max_idx] not in pred_unique:
                #     label[ins_mask>0] = 255            
            
            
        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                if hasattr(model.module.decode_head,
                           'debug_output_attention') and \
                        model.module.decode_head.debug_output_attention:
                    # Attention debug output
                    mmcv.imwrite(result[0] * 255, out_file)
                else:
                    model.module.show_result(
                        img_show,
                        result,
                        palette=dataset.PALETTE,
                        show=show,
                        out_file=out_file,
                        opacity=opacity)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result, tmpdir='.efficient_test')
            results.append(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def seco_single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    if efficient_test:
        mmcv.mkdir_or_exist('.efficient_test')
        
    predicted_label = np.zeros((len(data_loader), 256, 512))
    predicted_prob = np.zeros((len(data_loader), 256, 512))
    index = 0
    image_name = []
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, prob = model(return_loss=False, return_logit=True, **data)
            prob = F.interpolate(prob, size=([256, 512]), mode='bilinear', align_corners=True)
            batch_size, class_num = prob.shape[0], prob.shape[1]
            prob = prob.cpu().numpy()
            for bs_id in range(batch_size):
                prob_max = np.max(prob[bs_id], 0)
                pred_max = prob[bs_id].argmax(0)
                
                # uncomment the following line when visualizing SYNTHIA->Cityscapes # no for self training
                # pred = transform_color(pred_prob)
                
                predicted_label[index] = pred_max.copy()
                predicted_prob[index] = prob_max.copy() 
                
                img_metas = data['img_metas'][0].data[0]
                image_name.append(img_metas[bs_id]['ori_filename'])
                index += 1
        
        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                if hasattr(model.module.decode_head,
                           'debug_output_attention') and \
                        model.module.decode_head.debug_output_attention:
                    # Attention debug output
                    mmcv.imwrite(result[0] * 255, out_file)
                else:
                    model.module.show_result(
                        img_show,
                        result,
                        palette=dataset.PALETTE,
                        show=show,
                        out_file=out_file,
                        opacity=opacity)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result, tmpdir='.efficient_test')
            results.append(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    
    ### save top- images
    select_num_dict = {}
    thres = []
    for i in range(class_num):
        x = predicted_prob[predicted_label==i]
        if len(x) == 0:
            thres.append(0)
            continue        
        x = np.sort(x)
        thres.append(x[int(np.round(len(x)*0.50))])
    print(thres)
    thres = np.array(thres)
    #thres[thres>0.9]=0.9
    #print(thres)

    for index in range(len(image_name)):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        for i in range(class_num):
            label[(prob<thres[i])*(label==i)] = 255
        select_num_dict[name] = np.sum(np.asarray(label, dtype=np.uint8)!=255)
        mask = get_color_pallete(np.asarray(label, dtype=np.uint8), "city")
        output_folder = 'seco_save_masks'
        save_path = os.path.join(output_folder, name)
        print('mask save to', save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mask.save(save_path)    
    exp_name = 'HRDA_seco'
    psd_prefix_name = '{}/'.format(exp_name)
    image_prefix_name = 'leftImg8bit/train/'
    ranked_name_list = sorted(image_name, key=lambda c: select_num_dict[c], reverse=True)
    ranked_name_list = [ os.path.join(image_prefix_name + xx.replace('gtFine_labelTrainIds', 'leftImg8bit')) + ' ' + os.path.join(psd_prefix_name, xx) for xx in ranked_name_list]
    topK = int(len(ranked_name_list) * 0.2)

    labeled_save_path = 'splits/{}/labeled.txt'.format(exp_name)
    unlabeled_save_path = 'splits/{}/unlabeled.txt'.format(exp_name)
    os.makedirs(os.path.dirname(labeled_save_path), exist_ok=True)
    text_save(labeled_save_path, ranked_name_list[:topK])
    text_save(unlabeled_save_path, ranked_name_list[topK:])
    
    return results
    
def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    if efficient_test:
        mmcv.mkdir_or_exist('.efficient_test')
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result, tmpdir='.efficient_test')
            results.append(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results
