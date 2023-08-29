#!/bin/bash

#SBATCH --job-name x+a2      # 作业名
#SBATCH --partition compute
#SBATCH --nodes 1

#SBATCH --gres=gpu:1
#SBATCH --mem 16384

#SBATCH --output x+a2.log       # 保存输出结果

echo Start slurm job at `date`
# running
module load anaconda/3
module load cuda/10.2
module load cudnn/8.0.4-cuda10.2
module load gcc/5.4.0
source activate slowfast
export PYTHONPATH=/public/home/zhouz/perl5/HAR/slowfast:$PYTHONPATH
#export PYTHONPATH=/public/home/zhouz/perl5/Drone-TIP/slowfast/models:$PYTHONPATH
#export PYTHONPATH=/public/home/zhouz/perl5/Drone-TIP/slowfast/datasets:$PYTHONPATH
#python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml NUM_GPUS 1 TRAIN.BATCH_SIZE 16 SOLVER.BASE_LR 0.025 DATA.PATH_TO_DATA_DIR /public/home/jiaxm/perl5/datasets/ucf256/
#python setup.py build develop
#CUDA_VISIBLE_DEVICES=1
python tools/run_net.py --cfg /public/home/zhouz/perl5/HAR/configs/Kinetics/SLOW_8x8_R50.yaml DATA_LOADER.NUM_WORKERS 0 NUM_GPUS 1 \
TRAIN.BATCH_SIZE 8 SOLVER.BASE_LR 0.0005 SOLVER.MAX_EPOCH 60 SOLVER.WEIGHT_DECAY 1e-06 SOLVER.WARMUP_EPOCHS 0.0 \
DATA.PATH_TO_DATA_DIR /public/home/zhouz/perl5/dataset/crime/list3/

#python tools/run_net.py --cfg /public/home/zhouz/perl5/HAR/configs/Kinetics/SLOW_8x8_R50.yaml DATA_LOADER.NUM_WORKERS 0 NUM_GPUS 1 \
#TRAIN.BATCH_SIZE 8 SOLVER.BASE_LR 0.0005 SOLVER.WEIGHT_DECAY 1e-06 SOLVER.WARMUP_EPOCHS 0.0 \
#DATA.PATH_TO_DATA_DIR /public/home/zhouz/perl5/dataset/crime/list2/ \
#TEST.CHECKPOINT_FILE_PATH /public/home/zhouz/perl5/HAR/x+e/checkpoints/checkpoint_epoch_00050.pyth \
#TRAIN.ENABLE False \









echo End slurm job at `date`