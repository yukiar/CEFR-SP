#!/bin/bash
# Train from scratch
stage=0
stop_stage=1000
# data-related
score_names="content organization grammar vocabulary"
anno_fn="109年寫作語料.xlsx"
kfold=5
test_on_valid="true"
do_round="true"
trans_type="origin"
# model-related
model=bert
model_type=bert-model
model_path=bert-base-uncased
max_score=8
max_seq_length=512
max_epochs=-1
alpha=0.2
num_prototypes=3
monitor="val_loss"
model_type=contrastive
do_loss_weight=true
do_lower_case=true
init_lr=5.0e-5
batch_size=8
extra_options=""

. ./path.sh
. ./parse_options.sh
set -euo pipefail

data_dir=../data-writting/gsat109/$trans_type
exp_root=../exp-writting/gsat109/$trans_type
folds=`seq 1 $kfold`

if [ "$test_on_valid" == "true" ]; then
    data_dir=${data_dir}_tov
    exp_root=${exp_root}_tov
fi


if [ "$model_type" == "classification" ] || [ "$model_type" == "regression" ]; then
    exp_tag=level_estimator_${model_type}
else
    exp_tag=level_estimator_${model_type}_num_prototypes${num_prototypes}
fi

if [ "$do_loss_weight" == "true" ]; then
    exp_tag=${exp_tag}_loss_weight_alpha${alpha}
    extra_options="$extra_options --with_loss_weight"
fi

if [ "$do_lower_case" == "true" ]; then
    exp_tag=${exp_tag}_lcase
    extra_options="$extra_options --do_lower_case"
fi

if [ "$max_epochs" != "-1" ]; then
    exp_tag=${exp_tag}_ep${max_epochs}
fi

model_name=`echo $model_path | sed -e 's/\//-/g'`
exp_tag=${exp_tag}_${model_name}_${monitor}_b${batch_size}_lr${init_lr}



if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then  
     
    for sn in $score_names; do
        for fd in $folds; do
            exp_dir=$exp_tag/$sn/$fd
            python level_estimator.py --model $model_path --lm_layer 11 $extra_options \
                                      --CEFR_lvs  $max_score \
                                      --seed 985 --num_labels $max_score \
                                      --max_epochs $max_epochs \
                                      --monitor $monitor \
                                      --exp_dir $exp_dir \
                                      --score_name $sn \
                                      --batch $batch_size --warmup 0 \
                                      --num_prototypes $num_prototypes --type ${model_type} --init_lr $init_lr \
                                      --alpha $alpha --data $data_dir/$fd --test $data_dir/$fd --out $exp_root
        done
    done
fi



if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then  
    
    for sn in $score_names; do
        for fd in $folds; do
            # Test a pretrained model
            checkpoint_path=`find $exp_root/$exp_tag/$sn/$fd/version_0 -name *ckpt`
            
            if [ -d $exp_root/$exp_tag/$sn/$fd/version_1 ]; then
                rm -rf $exp_root/$exp_tag/$sn/$fd/version_1
            fi

            echo "$part $sn $fd"
            echo $checkpoint_path
            exp_dir=$exp_tag/$sn/$fd
            python level_estimator.py --model $model_path --lm_layer 11 $extra_options \
                                      --CEFR_lvs  $max_score \
                                      --seed 985 --num_labels $max_score \
                                      --max_epochs $max_epochs \
                                      --monitor $monitor \
                                      --exp_dir $exp_dir \
                                      --score_name $sn \
                                      --batch $batch_size --warmup 0 \
                                      --num_prototypes $num_prototypes --type ${model_type} --init_lr $init_lr \
                                      --alpha $alpha --data $data_dir/$fd --test $data_dir/$fd --out $exp_root --pretrained $checkpoint_path

        done
    done 
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then  
    runs_root=$exp_root
    python local/speaking_predictions_to_report.py  --data_dir $data_dir \
                                                    --result_root $runs_root/$exp_tag \
                                                    --folds "$folds" \
                                                    --version_dir version_0 \
                                                    --scores "$score_names" > $runs_root/$exp_tag/report.log
    
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then  
    runs_root=$exp_root
    echo $runs_root/$exp_tag
    python local/visualization.py   --result_root $runs_root/$exp_tag \
                                    --scores "$score_names"
fi
