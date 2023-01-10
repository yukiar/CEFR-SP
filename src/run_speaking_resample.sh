#!/bin/bash
# Train from scratch
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
n_resamples=100
# model-related
model=bert
exp_tag=bert-model
model_path=bert-base-uncased
max_score=8
max_seq_length=512
max_epochs=-1
alpha=0.2
monitor="train_loss"
extra_options=

. ./path.sh
. ./parse_options.sh

set -euo pipefail

data_dir=../data-speaking/gept-p${part}/$trans_type
exp_root=../exp-speaking/gept-p${part}/$trans_type
folds=`seq 1 $kfold`

if [ "$test_on_valid" == "true" ]; then
    extra_options="--test_on_valid"
    data_dir=${data_dir}_tov
    exp_root=${exp_root}_tov
fi

if [ "$do_round" == "true" ]; then
    extra_options="$extra_options --do_round"
    data_dir=${data_dir}_round
    exp_root=${exp_root}_round
fi

if [ "$merge_below_b1" == "true" ]; then
    extra_options="$extra_options --merge_below_b1"
    data_dir=${data_dir}_bb1
    exp_root=${exp_root}_bb1
fi

if [ "$max_epochs" != "-1" ]; then
    exp_tag=level_estimator_contrastive_loss_weight_num_prototypes${num_prototypes}_max_epochs${max_epochs}
else
    exp_tag=level_estimator_contrastive_loss_weight_num_prototypes${num_prototypes}
fi

model_name=`echo $model_path | sed -e 's/\//-/g'`
exp_tag=${exp_tag}_${model_name}_${monitor}_alpha$alpha

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then  
     
    for sn in $score_names; do
        for fd in $folds; do
            # model_args_dir
            new_data_dir=${data_dir}_r${n_resamples}_${sn}
            new_exp_root=${exp_root}_r${n_resamples}_${sn}
            exp_dir=$exp_tag/$sn/$fd
            
            python level_estimator.py --model $model_path --lm_layer 11 --do_lower_case \
                                      --CEFR_lvs  $max_score \
                                      --seed 985 --num_labels $max_score \
                                      --max_epochs $max_epochs \
                                      --monitor $monitor \
                                      --exp_dir $new_exp_dir \
                                      --score_name $sn \
                                      --batch 8 --warmup 0 --with_loss_weight \
                                      --num_prototypes $num_prototypes --type contrastive --init_lr 1.0e-5 \
                                      --alpha $alpha --data $new_data_dir/$fd --test $new_data_dir/$fd --out $new_exp_root
           
        done
    done
fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then  
    
    for sn in $score_names; do
        for fd in $folds; do
            new_data_dir=${data_dir}_r${n_resamples}_${sn}
            new_exp_root=${exp_root}_r${n_resamples}_${sn}
            exp_dir=$exp_tag/$sn/$fd
            # Test a pretrained model
            checkpoint_path=`find $new_exp_root/$exp_dir/version_0 -name *ckpt`
            echo "$part $sn $fd"
            echo $checkpoint_path
            python level_estimator.py --model $model_path --lm_layer 11 --do_lower_case \
                                      --CEFR_lvs  $max_score \
                                      --seed 985 --num_labels $max_score \
                                      --max_epochs $max_epochs \
                                      --monitor $monitor \
                                      --exp_dir $exp_dir \
                                      --score_name $sn \
                                      --batch 8 --warmup 0 --with_loss_weight \
                                      --num_prototypes $num_prototypes --type contrastive --init_lr 1.0e-5 \
                                      --alpha $alpha --data $new_data_dir/$fd --test $new_data_dir/$fd \
                                      --out $new_exp_root --pretrained $checkpoint_path
        done
    done 
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then  
    for sn in $score_names; do
        new_data_dir=${data_dir}_r${n_resamples}_${sn}
        new_runs_root=${exp_root}_r${n_resamples}_${sn}
        python local/speaking_predictions_to_report.py  --data_dir $new_data_dir \
                                                    --result_root $new_runs_root/$exp_tag \
                                                    --folds "$folds" \
                                                    --version_dir version_0 \
                                                    --scores "$score_names" > $new_runs_root/$exp_tag/report.log
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then  
    for sn in $score_names; do
        new_data_dir=${data_dir}_r${n_resamples}_${sn}
        new_runs_root=${exp_root}_r${n_resamples}_${sn}
        echo $new_runs_root/$exp_tag

        python local/visualization.py   --result_root $new_runs_root/$exp_tag \
                                        --scores "$sn"
    done
fi
