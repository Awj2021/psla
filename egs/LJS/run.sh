#!/bin/bash
#SBATCH -p sm
#SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-5
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2,9]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="psla_fsd"
#SBATCH --output=./log_%j.txt

set -x
echo "Start training using dataset of LJS..."
export TORCH_HOME=./

att_head=4
model=efficientnet
psla=True
eff_b=2
batch_size=8

if [ $psla == True ]
then
  impretrain=True
  freqm=48
  timem=192
  mixup=0
  bal=False
else
  impretrain=False
  freqm=0
  timem=0
  mixup=0
  bal=False
fi

lr=5e-4
trpath=/vol/research/wenjieProject/projects/audio/dataset/train_data.json

epoch=40
wa_start=21
wa_end=40
lrscheduler_start=10

exp_dir=./exp/demo-${model}-${eff_b}-${lr}-LJS-impretrain-${impretrain}-fm${freqm}-tm${timem}-mix${mixup}-b${batch_size}-2
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python ../../src/run.py --data-train $trpath --data-val /vol/research/wenjieProject/projects/audio/dataset/test_data.json \
--data-eval /vol/research/wenjieProject/projects/audio/dataset/test_data.json --exp-dir $exp_dir \
--n-print-steps 1000 --save_model True --num-workers 5 --label-csv ./class_labels_indices.csv \
--n_class 2 --n-epochs ${epoch} --batch-size ${batch_size} --lr $lr \
--model ${model} --eff_b $eff_b --impretrain ${impretrain} --att_head ${att_head} \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} --lr_patience 2 \
--dataset_mean -4.6476 --dataset_std 4.5699 --target_length 3000 --noise False \
--metrics mAP --warmup True --loss BCE --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay 0.5 \
--wa True --wa_start ${wa_start} --wa_end ${wa_end}
