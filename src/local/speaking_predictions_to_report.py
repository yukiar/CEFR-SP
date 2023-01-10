import argparse
import random
import logging
import os
import csv
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
from metrics_cpu import compute_metrics
import matplotlib.pyplot as plt

'''
因為run_speech_grader在--do_test是SequentialSampler(和data下的.tsv順序一致)
因此，我們取data/*.tsv的text_id，以及runs/bert_model/*/predictions.txt中的分數，組成回傳報告(i.e., runs/reports.csv)。

'''

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="data",
                    type=str)

parser.add_argument("--result_root",
                    default="runs/bert-model-writing",
                    type=str)

parser.add_argument("--folds",
                    default="1 2 3 4 5",
                    type=str)
                    
parser.add_argument("--tsv_fn",
                    default="test.tsv",
                    type=str)

parser.add_argument("--version_dir",
                    default="version_0",
                    type=str)

parser.add_argument("--scores",
                    default="content pronunciation vocabulary",
                    type=str)

args = parser.parse_args()


def filled_csv(csv_dict, result_root, score, nf, text_ids, version_dir):
     
    kfold_dir = os.path.join(result_root, score, nf, version_dir) 
    pred_path = os.path.join(kfold_dir, "predictions.txt")
    
    if not os.path.exists(pred_path):
        return False
    
    with open(pred_path, "r") as fn:
        for i, line in enumerate(fn.readlines()):
            text_id = text_ids[nf][i]
            pred_score, anno_score = line.split("|")
            pred_score = float(pred_score.split()[0]) + 1
            anno_score = float(anno_score.split()[0]) + 1
            
            assert csv_dict[nf][text_id]["anno"][score] == anno_score
            csv_dict[nf][text_id]["pred"][score] = pred_score
    return True


def evaluation(total_losses_score_nf, evaluate_dict, target_score="organization", np_bins=None):
    # 1. origin
    # MSE, PCC, within0.5, within1.0
    all_score_preds = []
    all_score_annos = []
    total_losses_score_nf["origin"] = {}
    
    for text_id, scores_info in evaluate_dict.items(): 
        pred_score = float(scores_info["pred"][target_score])
        anno_score = float(scores_info["anno"][target_score])
        all_score_preds.append(pred_score)
        all_score_annos.append(anno_score)
    
    all_score_preds = np.array(all_score_preds)
    all_score_annos = np.array(all_score_annos)
    
    if np_bins is not None:
        all_score_preds = np.digitize(all_score_preds, np_bins) + 1
        all_score_annos = np.digitize(all_score_annos, np_bins) + 1
    
    compute_metrics(total_losses_score_nf["origin"], all_score_preds, all_score_annos)
     
    return total_losses_score_nf, all_score_preds, all_score_annos


data_dir = args.data_dir
n_folds = args.folds.split()
result_root = args.result_root
scores = args.scores.split()
version_dir = args.version_dir

csv_header = "text_id " + " ".join(scores)
csv_header = csv_header.split()
csv_dict = {}
text_ids = {}

for nf in n_folds:
    text_ids[nf] = []
    csv_dict[nf] = defaultdict(dict)
    tsv_path = os.path.join(data_dir, nf, args.tsv_fn)

    with open(tsv_path, 'r') as fn:
        csv_reader = csv.reader(fn, delimiter='\t')
        
        for i, row in tqdm(enumerate(csv_reader)):
            if i == 0: 
                columns = {key:header_index for header_index, key in enumerate(row)}
                continue
                
            text_id, text = row[columns["text_id"]], row[columns["text"]]
             
            text_ids[nf].append(text_id)
            csv_dict[nf][text_id]["anno"] = { s: float(row[columns[s]]) for s in scores }        
            csv_dict[nf][text_id]["pred"] = { s: float(row[columns[s]]) for s in scores }

# fiiled csv_dict
total_losses = defaultdict(dict)
total_df_losses = defaultdict(dict)
average_losses = defaultdict(dict)
infos = ["anno", "anno(cefr)", "pred", "pred(cefr)"]

scores_ = []
for nf in n_folds:
    for score in scores:
        sucessful = filled_csv(csv_dict, result_root, score, nf, text_ids, version_dir)
        if sucessful:
            if score not in scores_:
                scores_.append(score)
             
scores = list(scores_)
kfold_info = {}

for score in scores:
    kfold_info[score] = {str(1+i):{info:[] for info in infos} for i in range(len(n_folds))}
    kfold_info[score]["All"] = {info:[] for info in infos}

print("ORIGIN")
print("scores", scores)

all_bins = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
for score in scores: 
    for nf in n_folds: 
        total_losses[score][nf] = {}
        total_losses[score][nf], all_score_preds, all_score_annos = evaluation(total_losses[score][nf], csv_dict[nf], score, all_bins)
        kfold_dir = os.path.join(result_root, score, nf) 
        kfold_info[score][nf]["anno"] += all_score_annos.tolist()
        kfold_info[score][nf]["pred"] += all_score_preds.tolist() 
        kfold_info[score]["All"]["anno"] += all_score_annos.tolist()
        kfold_info[score]["All"]["pred"] += all_score_preds.tolist()
        #print(nf, len(kfold_info[score][nf]["pred"]))
        #fig = csv_dict[nf].plot.scatter(figsize=(20, 16), fontsize=26).get_figure()
        
    ave_losses = {k:0 for k in list(total_losses[score][n_folds[0]]["origin"].keys())}
    df_losses = {k:[] for k in list(total_losses[score][n_folds[0]]["origin"].keys())}
    
    for nf in n_folds: 
        for metric in list(total_losses[score][nf]["origin"].keys()):
            ave_losses[metric] += 1/len(n_folds) * total_losses[score][nf]["origin"][metric]
            df_losses[metric].append(total_losses[score][nf]["origin"][metric])

    average_losses[score] = ave_losses
    print(score, ave_losses)
    df_losses = pd.DataFrame.from_dict(df_losses)
    print(df_losses.mean())

print()
print("CEFR")
cefr_bins = np.array([2.5, 4.5, 6.5])

for score in scores:
    
    for nf in n_folds: 
        total_losses[score][nf] = {}
        total_losses[score][nf], all_score_preds, all_score_annos = evaluation(total_losses[score][nf], csv_dict[nf], score, cefr_bins)
        kfold_dir = os.path.join(result_root, score, nf) 
        
        kfold_info[score][nf]["anno(cefr)"] += all_score_annos.tolist()
        kfold_info[score][nf]["pred(cefr)"] += all_score_preds.tolist() 
        kfold_info[score]["All"]["anno(cefr)"] += all_score_annos.tolist()
        kfold_info[score]["All"]["pred(cefr)"] += all_score_preds.tolist()
        
    result_dir = os.path.join(result_root, score)
    with pd.ExcelWriter(os.path.join(result_dir, "kfold_detail.xlsx")) as writer:
        for f in list(kfold_info[score].keys()):
            df = pd.DataFrame(kfold_info[score][f])
            df.to_excel(writer, sheet_name=f)   
    
    ave_losses = {k:0 for k in list(total_losses[score][n_folds[0]]["origin"].keys())}
    df_losses = {k:[] for k in list(total_losses[score][n_folds[0]]["origin"].keys())}
        
    for nf in n_folds: 
        for metric in list(total_losses[score][nf]["origin"].keys()):
            ave_losses[metric] += 1/len(n_folds) * total_losses[score][nf]["origin"][metric]
            df_losses[metric].append(total_losses[score][nf]["origin"][metric])

    average_losses[score] = ave_losses
    print(score, ave_losses)
    df_losses = pd.DataFrame.from_dict(df_losses)
    print(df_losses.mean())
