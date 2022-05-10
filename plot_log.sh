#!/bin/bash

recent=$(ls -t *.log |head -1)
echo $recent
log_folder="./grep_log"
if [ ! -d $log_folder ]; then
    mkdir $log_folder
fi
#grep " trainval-acc@" $recent |awk '{print $7, $9, $11}' > $log_folder/train_acc
#grep " val-acc@" $recent |awk '{print $7, $9, $11}'> $log_folder/val_acc
#grep " loss=" $recent |awk '{print $6, $8, $10, $12, $14, $16, $18}' > $log_folder/loss

grep "noraml" $recent |awk '{print $7, $9, $11}' > $log_folder/train_acc
grep "all\ eval" $recent |awk '{print $7, $9, $11}'> $log_folder/val_acc
grep "as\ eval" $recent |awk '{print $7, $9, $11}'> $log_folder/asval_acc

python utils/plot_log.py --dir $log_folder


