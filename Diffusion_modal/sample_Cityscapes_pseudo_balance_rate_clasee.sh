CUDA_VISIBLE_DEVICES=1 nohup python LIS.py --batch_size 10 \
                                           --out_num 1000 \
                                             --config configs/stable-diffusion/v1-finetune_Cityscapes.yaml \
					     --ckpt logs/2023-12-17T18-34-30_exp_cityscapes_masked_pseudo_rate_class/checkpoints/last.ckpt \
                                             --dataset CityscapesBalance \
					     --outdir outputs/Cityscapes_LIS_mask_pseudo_balance_rate_class \
                                             --txt_file /data/1_2/1_2.p  \
                                             --data_root /data/seco/ \
                                             --plms > logs/LIS_hard_pseudo_balance_rate_class.logs 2>&1 &
