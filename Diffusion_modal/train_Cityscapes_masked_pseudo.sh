CUDA_VISIBLE_DEVICES=0 nohup python main.py --base configs/stable-diffusion/v1-finetune_Cityscapes.yaml \
                                      -t \
                                      --actual_resume models/ldm/stable-diffusion/sd-v1-4-full-ema.ckpt \
                                      -n exp_cityscapes_masked_pseudo_rate_class \
                                      --gpus 0, \
                                      --data_root /data/UniMatch-main/dataset/Cityscapes/ \
                                      --train_txt_file /data/splits/cityscapes/DTST_DIFF/rate_class.txt \
                                      --val_txt_file /data/DTST/datasets/cityscapes_val_list.txt \
				      > logs/cityscapes_train_pseudo.file 2>&1 &
