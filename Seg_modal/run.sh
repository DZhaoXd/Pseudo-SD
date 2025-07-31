
CUDA_VISIBLE_DEVICES="0,1,2,3" nohup sh ./scripts/train_synthia.sh 4 2023 > logs/seco_synthia.log 2>&1 &

CUDA_VISIBLE_DEVICES="1" nohup  python Eval.py --config=./configs/cityscapes.yaml --labeled-id-path splits/cityscapes/train.txt --resume-path ./exp/unimatch/r101/seco_filter_64.6/ > logs/eval.log 2>&1 &

