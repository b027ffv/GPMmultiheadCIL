#!/bin/bash

# method: GPCNS, GPM, SGP
method='GPM'

# ================================================
d='cifar100-10'
s=0
gpu=0

# cifar100-10 cifar100-20;
# For baseline
python main_cifar100.py --seed $s --cuda $gpu --method $method --dataset $d --lr 0.05 --lr_min 5e-5 --scale_coff 5 --exp F_${d}_${method}_s${s}
# For Ours
python main_cifar100.py --seed $s --cuda $gpu --method $method --dataset $d --lr 0.05 --lr_min 5e-5 --lr2 0.05 --lr2_min 5e-5 --scale_coff 5 --merge_list 200 --exp F_${d}_${method}_FM_s${s}

# ================================================
d='miniimagenet'
s=0
gpu=0
# miniimagenet
# For baseline
python main_cifar100.py --seed $s --cuda $gpu --method $method --dataset $d --lr 0.1 --lr_min 1e-3 --lr_patience 5 --lr_factor 3 --n_epochs 200 --scale_coff 3 --gpm_eps 0.98 --gpm_eps_inc 0.001 --exp F_${d}_${method}_s${s} 
# For Ours
python main_cifar100.py --seed $s --cuda $gpu --method $method --dataset $d --lr 0.1 --lr_min 1e-3 --lr_patience 5 --lr_factor 3 --n_epochs 200 --scale_coff 3 --gpm_eps 0.98 --gpm_eps_inc 0.001 --lr2 0.1 --lr2_min 1e-3 --merge_list 200 --exp F_${d}_${method}_FM_s${s} 
