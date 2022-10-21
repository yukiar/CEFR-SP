from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from util import mean_pooling
import numpy as np
import tqdm


def read_cefr_corpus(corpus_path):
    levels, sents = [], []
    lv_indices = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    with open(corpus_path) as f:
        all_texts = f.readlines()
    for line in all_texts[1:]:  # skip the header
        array = line.strip().split('\t')
        annotator_x = int(array[2])
        annotator_y = int(array[4])
        if abs(annotator_x - annotator_y) <= 1:
            lv = max(annotator_x, annotator_y) - 1  # To make a label 0-base
            levels.append(lv)
            sents.append(array[1])
            lv_indices[lv].append(len(sents) - 1)

    return np.array(levels), sents, lv_indices


def distance_matrix(tokenizer, model, all_sentences, batch_size=64):
    all_sentence_embeddings = torch.zeros((len(all_sentences), model.config.hidden_size))

    for bidx in tqdm.tqdm(range(0, len(all_sentences), batch_size)):
        sentences = all_sentences[bidx:bidx + batch_size]
        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        encoded_input = encoded_input.to(model.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        sentence_embeddings = mean_pooling(token_embeddings, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        all_sentence_embeddings[bidx:bidx + batch_size] = sentence_embeddings

    matrix = 1 - torch.mm(all_sentence_embeddings, all_sentence_embeddings.t())
    matrix = matrix.cpu().detach().numpy()

    return matrix


def out_file(out_path, dataset):
    with open(out_path, 'w') as fw:
        for lv, sentences in dataset.items():
            for sent in sentences:
                fw.write('{0}\t{1}\n'.format(lv, sent))


def check_number_of_cases(case_per_level, dataset):
    for lv, cases in case_per_level.items():
        assert len(dataset[lv]) == cases


if __name__ == '__main__':
    # ##### Needs Manual setting: CEFR #####
    out_path_prefix = '../data/CEFR_level_annotation_'
    levels, sentences, lv_indices = read_cefr_corpus('../data/CEFR_level_annotation_full20k.txt')
    devcase_per_level = {0: int(len(lv_indices[0]) * 0.3), 1: int(len(lv_indices[1]) * 0.1),
                         2: int(len(lv_indices[2]) * 0.1), 3: int(len(lv_indices[3]) * 0.1),
                         4: int(len(lv_indices[4]) * 0.1), 5: int(len(lv_indices[5]) * 0.3)}
    testcase_per_level = {0: int(len(lv_indices[0]) * 0.3), 1: int(len(lv_indices[1]) * 0.1),
                          2: int(len(lv_indices[2]) * 0.1), 3: int(len(lv_indices[3]) * 0.1),
                          4: int(len(lv_indices[4]) * 0.1), 5: int(len(lv_indices[5]) * 0.3)}
    traincase_per_level = {0: len(lv_indices[0]) - devcase_per_level[0] - testcase_per_level[0],
                           1: len(lv_indices[1]) - devcase_per_level[1] - testcase_per_level[1],
                           2: len(lv_indices[2]) - devcase_per_level[2] - testcase_per_level[2],
                           3: len(lv_indices[3]) - devcase_per_level[3] - testcase_per_level[3],
                           4: len(lv_indices[4]) - devcase_per_level[4] - testcase_per_level[4],
                           5: len(lv_indices[5]) - devcase_per_level[5] - testcase_per_level[5]}
    # ##########

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to('cuda')

    dmatrix = distance_matrix(tokenizer, model, sentences)
    # make self-distance large
    for i in range(len(sentences)):
        dmatrix[i, i] = 999
    mindists = np.min(dmatrix, axis=0)
    mean_mindist = np.mean(mindists)

    test_sets = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    dev_sets = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    train_sets = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    sentences = np.array(sentences)
    for lv, indices in lv_indices.items():
        indices = np.array(indices)
        lv_mindists = mindists[indices]
        sort_indices = np.argsort(lv_mindists)[::-1]
        assert lv_mindists[sort_indices[testcase_per_level[lv] - 1]] >= mean_mindist
        # assert lv_mindists[sort_indices[testcase_per_level[lv] - 1 + devcase_per_level[lv] - 1]] >= mean_mindist
        test_sets[lv] = sentences[indices[sort_indices[0:testcase_per_level[lv]]]]
        dev_sets[lv] = sentences[
            indices[sort_indices[testcase_per_level[lv]:testcase_per_level[lv] + devcase_per_level[lv]]]]
        train_sets[lv] = sentences[indices[sort_indices[testcase_per_level[lv] + devcase_per_level[lv]:]]]

    # Check results
    check_number_of_cases(testcase_per_level, test_sets)
    check_number_of_cases(devcase_per_level, dev_sets)
    check_number_of_cases(traincase_per_level, train_sets)

    # Output
    out_file(out_path_prefix + 'train.txt', train_sets)
    out_file(out_path_prefix + 'dev.txt', dev_sets)
    out_file(out_path_prefix + 'test.txt', test_sets)
    print('Done!')
