#!/bin/bash

# ================================================
d='cifar100-10'
s=0
gpu=0

# cifar100-10 cifar100-20;
# For baseline
python main_cifar100_trgp.py --seed $s --cuda $gpu --method TRGP --dataset $d --exp F_${d}_TRGP_s${s}
# For Ours
python main_cifar100_trgp.py --seed $s --cuda $gpu --method TRGP --dataset $d --merge_list 200 --exp F_${d}_TRGP_FM_s${s}


# ================================================
d='miniimagenet'
s=0
gpu=0
# miniimagenet
# For baseline
python main_cifar100_trgp.py --seed $s --cuda $gpu --method TRGP --dataset $d --lr 0.1 --lr_min 1e-3 --lr_patience 5 --lr_factor 3 --n_epochs 100 --exp F_${d}_TRGP_s${s} 
# For Ours
python main_cifar100_trgp.py --seed $s --cuda $gpu --method TRGP --dataset $d --lr 0.1 --lr_min 1e-3 --lr_patience 5 --lr_factor 3 --n_epochs 100 --lr2 0.1 --lr2_min 1e-3 --merge_list 100 --exp F_${d}_TRGP_FM_s${s} 
