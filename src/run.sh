#!/bin/bash
# Train from scratch
python level_estimator.py --model bert-base-cased --lm_layer 11 --seed 935 --num_labels 6 --batch 128 --warmup 0 --with_loss_weight --num_prototypes 3 --type contrastive --init_lr 1.0e-5 --alpha 0.2 --data ../CEFR-SP/SCoRE/CEFR-SP_SCoRE_ --test ../CEFR-SP/SCoRE/CEFR-SP_SCoRE_ --out ../out/

# Test a pretrained model
python level_estimator.py --model bert-base-cased --lm_layer 11 --seed 935 --num_labels 6 --batch 128 --warmup 0 --with_loss_weight --num_prototypes 3 --type contrastive --init_lr 1.0e-5 --alpha 0.2 --data ../CEFR-SP/SCoRE/CEFR-SP_SCoRE_ --test ../CEFR-SP/SCoRE/CEFR-SP_SCoRE_ --out ../out/ --pretrained ../path_to_your_model.ckpt

