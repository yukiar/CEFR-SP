import torch, scipy
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        self.data_len=len(encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.data_len

class CEFRDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, sent_levels_a, sent_levels_b):
        self.encodings = encodings
        self.slabels_low = np.minimum(sent_levels_a, sent_levels_b)
        self.slabels_high = np.maximum(sent_levels_a, sent_levels_b)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['slabels_low'] = self.slabels_low[idx].clone().detach()
        item['slabels_high'] = self.slabels_high[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.slabels_high)


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def read_corpus(path, num_labels):
    levels_a, levels_b, sents = [], [], []
    with open(path) as f:
        for line in f:
            array = line.strip().split('\t')
            sents.append(array[0].split(' '))
            levels_a.append(float(array[1]) - 1)  # Convert 1-6 to 0-5
            levels_b.append(float(array[2]) - 1)  # Convert 1-6 to 0-5

    levels_a = np.array(levels_a)
    levels_b = np.array(levels_b)

    return levels_a, levels_b, sents


def convert_numeral_to_six_levels(levels):
    level_thresholds = np.array([0.0, 0.5, 1.5, 2.5, 3.5, 4.5])
    return _conversion(level_thresholds, levels)


def _conversion(level_thresholds, values):
    thresh_array = np.tile(level_thresholds, reps=(values.shape[0], 1))
    array = np.tile(values, reps=(1, level_thresholds.shape[0]))
    levels = np.maximum(np.zeros((values.shape[0], 1)),
                        np.count_nonzero(thresh_array <= array, axis=1, keepdims=True) - 1).astype(int)
    return levels


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Take attention mask into account for excluding padding
def token_embeddings_filtering_padding(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return token_embeddings * input_mask_expanded


def eval_multiclass(out_path, labels, predictions):
    cm = confusion_matrix(labels, predictions)
    plt.figure()
    sns.heatmap(cm)
    plt.savefig(out_path + '_confusion_matrix.png')

    report = classification_report(labels, predictions, digits=4)
    print(report)
    with open(out_path + '_test_report.txt', 'w') as fw:
        fw.write('{0}\n'.format(report))


def mean_confidence_interval(data, confidence=0.95):
    if len(data) > 5:
        data.remove(max(data))
        data.remove(min(data))
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h
