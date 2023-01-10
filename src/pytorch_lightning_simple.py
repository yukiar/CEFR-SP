"""
Optuna example that optimizes multi-layer perceptrons using PyTorch Lightning.
In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch Lightning, and FashionMNIST. We optimize the neural network architecture. As it is too time
consuming to use the whole FashionMNIST dataset, we here use a small subset of it.
You can run this example as follows, pruning can be turned on and off with the `--pruning`
argument.
    $ python pytorch_lightning_simple.py [--pruning]
"""
import argparse
import os
from typing import List
from typing import Optional

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
from util import eval_multiclass, read_corpus, convert_numeral_to_eight_levels
from model import LevelEstimaterClassification, LevelEstimaterContrastive, LevelEstimaterContrastivePcc
from baseline import BaselineClassification
from model_base import CEFRDataset

if version.parse(pl.__version__) < version.parse("1.0.2"):
    raise RuntimeError("PyTorch Lightning>=1.0.2 is required for this example.")

PERCENT_VALID_EXAMPLES = 0.1
EPOCHS = 10
DIR = os.getcwd()

def get_args():
    parser = argparse.ArgumentParser(description='CEFR level estimator.')
    parser.add_argument('--out', help='output directory', type=str, default='../optuna/icnale/trans_stt_whisperv2_large/1')
    parser.add_argument('--data', help='dataset', type=str, default="../data-speaking/icnale/trans_stt_whisperv2_large/1")
    parser.add_argument('--test', help='dataset', type=str, default="../data-speaking/icnale/trans_stt_whisperv2_large/1")
    parser.add_argument('--num_labels', help='number of attention heads', type=int, default=8)
    parser.add_argument('--alpha', help='weighing factor', type=float, default=0.2)
    parser.add_argument('--num_prototypes', help='number of prototypes', type=int, default=3)
    parser.add_argument('--model', help='Pretrained model', type=str, default='bert-base-cased')
    parser.add_argument('--pretrained', help='Pretrained level estimater', type=str, default=None)
    parser.add_argument('--with_loss_weight', action='store_true')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--lm_layer', help='number of attention heads', type=int, default=-1)
    parser.add_argument('--batch', help='Batch size', type=int, default=8)
    parser.add_argument('--seed', help='number of attention heads', type=int, default=42)
    parser.add_argument('--init_lr', help='learning rate', type=float, default=1e-5)
    parser.add_argument('--val_check_interval', help='Number of steps per validation', type=float, default=1.0)
    parser.add_argument('--warmup', help='warmup steps', type=int, default=0)
    parser.add_argument('--max_epochs', help='maximum epcohs', type=int, default=-1)
    ##### The followings are unused arguments: You can just ignore #####
    parser.add_argument('--beta', help='balance between sentence and word loss', type=float, default=0.5)
    parser.add_argument('--ib_beta', help='beta for information bottleneck', type=float, default=1e-5)
    parser.add_argument('--word_num_labels', help='number of attention heads', type=int, default=4)
    parser.add_argument('--CEFR_lvs', help='number of CEFR levels', type=int, default=8)
    parser.add_argument('--score_name', help='score_name for predict and train', type=str, default="holistic")
    parser.add_argument('--with_ib', action='store_true')
    parser.add_argument('--attach_wlv', action='store_true')
    parser.add_argument('--monitor', default='val_loss', type=str)
    parser.add_argument('--exp_dir', default='', type=str)
    
    ####################################################################
    args = parser.parse_args()
    args.attach_ib = True
    args.attach_wlv = True
    args.do_lower_case = True
    args.with_loss_weight = True
    return args


def objective(trial: optuna.trial.Trial) -> float:

    # We optimize the number of layers, hidden units in each layer and dropouts.
    num_prototypes = trial.suggest_int("num_prototypes", 2, 8)
    word_num_labels = trial.suggest_int("word_num_labels", 2, 8)
    alpha = trial.suggest_float("alpha", 0.1, 0.8)
    ib_beta = trial.suggest_float("ib_beta", 1e-6, 5e-3)
    init_lr = trial.suggest_float("init_lr", 1e-6, 5e-3)
    
    args = get_args()
    args.max_seq_length = trial.suggest_int("max_seq_length", 128, 512, 128)

    model = LevelEstimaterContrastive(args.data, args.test, args.model, 'contrastive', args.with_ib,
                                      args.with_loss_weight, args.attach_wlv,
                                      args.num_labels,
                                      word_num_labels,
                                      num_prototypes,
                                      alpha, ib_beta, args.batch,
                                      init_lr,
                                      args.warmup,
                                      args.lm_layer, args)

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
    )
    hyperparameters = dict(num_prototypes=num_prototypes, word_num_labels=word_num_labels,
                           alpha=alpha, ib_beta=ib_beta, init_lr=init_lr)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
