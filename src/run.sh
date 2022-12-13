#!/bin/bash
# Train from scratch
. ./path.sh
stage=0
stop_stage=1000
# data-related
score_names="content pronunciation vocabulary"
kfold=5
part=3
test_on_valid="true"
merge_below_b1="false"
trans_type="trans_stt"
do_round="true"
# model-related
model=bert
model_type=bert-model
model_path=bert-base-uncased
max_score=8
max_seq_length=512
num_epochs=6
extra_options=

. ./path.sh
. ./parse_options.sh

set -euo pipefail

data_dir=../data-speaking/gept-p${part}/$trans_type
exp_root=../exp-speaking/gept-p${part}/$trans_type
runs_root=runs-speaking/gept-p${part}/$trans_type
folds=`seq 1 $kfold`

if [ "$test_on_valid" == "true" ]; then
    extra_options="--test_on_valid"
    data_dir=${data_dir}_tov
    exp_root=${exp_root}_tov
    runs_root=${runs_root}_tov
fi

if [ "$do_round" == "true" ]; then
    extra_options="$extra_options --do_round"
    data_dir=${data_dir}_round
    exp_root=${exp_root}_round
    runs_root=${runs_root}_round
fi

if [ "$merge_below_b1" == "true" ]; then
    extra_options="$extra_options --merge_below_b1"
    data_dir=${data_dir}_bb1
    exp_root=${exp_root}_bb1
    runs_root=${runs_root}_bb1
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then  
     
    for sn in $score_names; do
        for fd in $folds; do
            echo "$part $sn $fd"
            python level_estimator.py --model $model_path --lm_layer 11 --do_lower_case \
                                     --seed 985 --num_labels $max_score \
                                     --score_name $sn \
                                     --batch 8 --warmup 0 --with_loss_weight \
                                     --num_prototypes 3 --type contrastive --init_lr 1.0e-5 \
                                     --fold_type $sn/$fd \
                                     --alpha 0.2 --data $data_dir/$fd --test $data_dir/$fd --out $exp_root
        done
    done
fi



if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then  
    
    for sn in $score_names; do
        for fd in $folds; do
            # Test a pretrained model
            checkpoint_path=`find $exp_root/*/$sn/$fd -name *ckpt`
            echo "$part $sn $fd"
            echo $checkpoint_path
            python level_estimator.py --model $model_path --lm_layer 11 --do_lower_case \
                                      --seed 985 --num_labels $max_score \
                                      --score_name $sn \
                                      --batch 8 --warmup 0 --with_loss_weight \
                                      --num_prototypes 3 --type contrastive --init_lr 1.0e-5 \
                                      --fold_type $sn/$fd \
                                      --alpha 0.2 --data $data_dir/$fd --test $data_dir/$fd --out $exp_root --pretrained $checkpoint_path


        done
    done 
fi



