# CEFR-Based Sentence Difficulty Annotation and Assessment

This directory contains codes to replicate our CEFR-level assessment model described in (Arase et al. 2022).

## Train level assessment model
Please specify your data directories with `data` and `test` arguments (the codes automatically add suffixes `_train.txt`, `_dev.txt`, and `_test.txt`).

In the paper, we concatenated the training, validation, and test sets of all sources (i.e., Newsela-Auto, Wiki-Auto, and SCoRE) and used for training and evaluation.

```
python level_estimator.py --model bert-base-cased --lm_layer 11 --seed 935 --num_labels 6 --batch 128 --warmup 0 --with_loss_weight --num_prototypes 3 --type contrastive --init_lr 1.0e-5 --alpha 0.2 --data ../CEFR-SP/SCoRE/CEFR-SP_SCoRE --test ../CEFR-SP/SCoRE/CEFR-SP_SCoRE --out ../out/
```

## Level assessment with a pretrained model
- Please specify your data directories with `data` and `test` arguments (the codes automatically add suffixes `_train.txt`, `_dev.txt`, and `_test.txt`).
- Please specify the path to your pretrained model at `pretrained`.
```
python level_estimator.py --model bert-base-cased --lm_layer 11 --seed 935 --num_labels 6 --batch 128 --warmup 0 --with_loss_weight --num_prototypes 3 --type contrastive --init_lr 1.0e-5 --alpha 0.2 --data ../CEFR-SP/SCoRE/CEFR-SP_SCoRE --test ../CEFR-SP/SCoRE/CEFR-SP_SCoRE --out ../out/ --pretrained ../path_to_your_model.ckpt
```
### Pretrained model
The pretrained model is available at [Zenodo](https://doi.org/10.5281/zenodo.7234096).

## Citation
Please cite the following paper if you use the above resources for your research.
 ```
 Yuki Arase, Satoru Uchida, and Tomoyuki Kajiwara. 2022. CEFR-Based Sentence-Difficulty Annotation and Assessment. 
 in Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2022) (Dec. 2022).
 
@inproceedings{arase:emnlp2022,
    title = "{CEFR}-Based Sentence-Difficulty Annotation and Assessment",
    author = "Arase, Yuki  and Uchida, Satoru, and Kajiwara, Tomoyuki",
    booktitle = "Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2022)",
    month = dec,
    year = "2022",
}
 ```

