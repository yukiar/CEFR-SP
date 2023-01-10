import numpy as np
import scipy.stats as stats


def compute_metrics(total_losses, all_score_predictions, all_score_targets):
    """ Computes Pearson correlation and accuracy within 0.5 and 1 of target score and adds each to total_losses dict. """
    total_losses['rmse'] = compute_rmse(all_score_predictions, all_score_targets)
    total_losses['mcrmse'] = compute_mcrmse(all_score_predictions, all_score_targets)
    total_losses['pearson'] = stats.pearsonr(all_score_predictions, all_score_targets)[0]
    total_losses['within_0.5'] = _accuracy_within_margin(all_score_predictions, all_score_targets, 0.5)
    total_losses['within_1'] = _accuracy_within_margin(all_score_predictions, all_score_targets, 1)
    total_losses['mcwithin_0.5'] = compute_within_acc(all_score_predictions, all_score_targets, 0.5)
    total_losses['mcwithin_1'] = compute_within_acc(all_score_predictions, all_score_targets, 1)

def _accuracy_within_margin(score_predictions, score_target, margin):
    """ Returns the percentage of predicted scores that are within the provided margin from the target score. """
    return np.sum(
        np.where(
            np.abs(score_predictions - score_target) <= margin,
            np.ones(len(score_predictions)),
            np.zeros(len(score_predictions)))).item() / len(score_predictions) * 100

def compute_rmse(all_score_predictions, all_score_targets):
    return np.sqrt(np.mean((all_score_predictions - all_score_targets)**2))

def compute_mcrmse(all_score_predictions, all_score_targets):
    unique_classes = np.unique(all_score_targets)
    num_classes = len(unique_classes)
    score_rmse = 0.
    
    for c in unique_classes:
        indices = np.where(all_score_targets == c)
        score_predictions = all_score_predictions[indices]
        score_targets = all_score_targets[indices]
        score_rmse += 1 / num_classes * compute_rmse(score_predictions, score_targets)
    
    return score_rmse

def compute_within_acc(all_score_predictions, all_score_targets, margin):
    unique_classes = np.unique(all_score_targets)
    num_classes = len(unique_classes)
    score_rmse = 0.
    
    for c in unique_classes:
        indices = np.where(all_score_targets == c)
        score_predictions = all_score_predictions[indices]
        score_targets = all_score_targets[indices]
        score_rmse += 1 / num_classes * _accuracy_within_margin(score_predictions, score_targets, margin)
    
    return score_rmse
