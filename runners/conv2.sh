#!/bin/bash
# specify a partition
#SBATCH -p bigbatch

# specify number of nodes
#SBATCH -N 1
# specify the wall clock time limit for the job hh:mm:ss
#SBATCH -t 72:00:00
# specify the job name
#SBATCH -J conv2
# specify the filen1ame to be used for writing output
#SBATCH -o /home-mscluster/mkruger/GitHub/cgiar-zindi-crop-classification/logs/conv2_fcmae_in22k_in1k_384.%N.%j.out
# specify the filename for stderr
#SBATCH -e /home-mscluster/mkruger/GitHub/cgiar-zindi-crop-classification/logs/conv2_fcmae_in22k_in1k_384.%N.%j.err


source ~/.bashrc
# cd /home-mscluster/mkruger/RoboCup
conda activate cgiar
cd /home-mscluster/mkruger/GitHub/cgiar-zindi-crop-classification
python3 -u matthew_main.py --model_name "convnextv2_base.fcmae_ft_in22k_in1k_384"&
wait;