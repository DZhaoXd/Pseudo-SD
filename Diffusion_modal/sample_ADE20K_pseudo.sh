CUDA_VISIBLE_DEVICES=6 python LIS.py --batch_size 2 \
                                             --config configs/stable-diffusion/v1-finetune_ADE20K.yaml \
                                             --ckpt ./pretrain/freestyle-sd-v1-4-ade20k.ckpt \
                                             --dataset ADE20K \
					     --seed 1000 \
                                             --outdir outputs/ADE20K_LIS_pseudo \
                                             --txt_file /data2/yjy/semantic-segmentation-pytorch/only_infer.txt \
                                             --data_root /data2/yjy/semantic-segmentation-pytorch \
                                             --plms 
