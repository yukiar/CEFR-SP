import glob, os
import numpy as np
from sklearn.metrics import cohen_kappa_score
from extract_eval_results import get_version, compute_conf_intervals


def eval_cohen_kappa(filepath, gold_H, gold_L):
    version = get_version(filepath)
    predictions = []
    with open(filepath) as f:
        if 'level_estimator_' in filepath:
            f.readline()
            for line in f:
                predictions.append(int(line.strip().replace('[', '').replace(']', '')))
        else:
            for line in f:
                predictions.append(int(line.strip()))
    predictions = np.array(predictions)
    assert predictions.shape == gold_H.shape

    gold = get_gold_labels(predictions, gold_H, gold_L)
    k = cohen_kappa_score(predictions, gold, weights='quadratic')

    return version, k


def get_gold_labels(predictions, lower_labels, higher_labels):
    if np.sum(predictions == lower_labels) >= np.sum(predictions == higher_labels):
        gold_labels = lower_labels
        gold_labels[predictions == higher_labels] = higher_labels[predictions == higher_labels]
    else:
        gold_labels = higher_labels
        gold_labels[predictions == lower_labels] = lower_labels[predictions == lower_labels]
    return gold_labels


def load_gold_labels(filepath):
    labels_H, labels_L = [], []
    with open(filepath) as f:
        for line in f:
            array = line.split('\t')
            labels_H.append(max(int(array[1]), int(array[2])))
            labels_L.append(min(int(array[1]), int(array[2])))

    return np.array(labels_H), np.array(labels_L)


if __name__ == '__main__':
    test_file = '../data/path_to_your_test_file.txt'
    out_dir = '../path_to_your_model_output_dir/'
    confidence = 0.95

    labels_H, labels_L = load_gold_labels(test_file)

    # Process test scores
    pred_file_paths = sorted(glob.glob(os.path.join(out_dir, '**/', 'test_predictions.txt'), recursive=True))

    results = {}
    for filepath in pred_file_paths:
        version, score = eval_cohen_kappa(filepath, labels_H, labels_L)
        if version not in results:
            results[version] = list()
        results[version].append(float(score))

    print('### Cohen Kappa ###')
    compute_conf_intervals(results, confidence)
