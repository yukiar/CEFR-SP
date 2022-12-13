import torch, transformers
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
from util import mean_pooling, token_embeddings_filtering_padding, read_corpus, CEFRDataset, eval_multiclass

class LevelEstimaterBase(pl.LightningModule):
    def __init__(self, corpus_path, test_corpus_path, pretrained_model, with_ib, attach_wlv,
                 num_labels,
                 word_num_labels, alpha,
                 batch_size,
                 learning_rate, warmup,
                 lm_layer,
                 args):
        super().__init__()
        self.save_hyperparameters()
        self.CEFR_lvs = args.CEFR_lvs

        if attach_wlv and with_ib:
            raise Exception('Information bottleneck and word labels cannot be used together!')

        self.corpus_path = corpus_path
        self.test_corpus_path = test_corpus_path
        self.pretrained_model = pretrained_model
        self.with_ib = with_ib
        self.attach_wlv = attach_wlv
        self.num_labels = num_labels
        self.word_num_labels = word_num_labels
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup = warmup
        self.lm_layer = lm_layer
        self.score_name = args.score_name
        self.do_lower_case = args.do_lower_case
        self.max_seq_length = 450
        self.special_tokens_count = 0

        # Load pre-trained model
        self.load_pretrained_lm()

    def load_pretrained_lm(self):
        if 'roberta' in self.pretrained_model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model, add_prefix_space=True, do_lower_case=self.do_lower_case, max_length=self.max_seq_length, truncation=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model, do_lower_case=self.do_lower_case, max_length=self.max_seq_length, truncation=True)
        self.lm = AutoModel.from_pretrained(self.pretrained_model)

    def precompute_loss_weights(self, epsilon=1e-5):
        train_levels_a, train_levels_b, _ = read_corpus(self.corpus_path + '/train.tsv', self.num_labels, self.score_name)
        train_levels = np.concatenate((train_levels_a, train_levels_b[train_levels_b != train_levels_a]))

        train_sentlv_ratio = np.array([np.sum(train_levels == lv) for lv in range(self.CEFR_lvs)])
        train_sentlv_ratio = train_sentlv_ratio / np.sum(train_sentlv_ratio)
        train_sentlv_weights = np.power(train_sentlv_ratio, self.alpha) / np.sum(
            np.power(train_sentlv_ratio, self.alpha)) / (train_sentlv_ratio + epsilon)

        return torch.Tensor(train_sentlv_weights)

    def encode(self, batch):
        outputs = self.lm(batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
        return outputs.hidden_states[self.lm_layer], None

    def forward(self, inputs):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def get_gold_labels(self, predictions, lower_labels, higher_labels):
        if torch.sum(predictions == lower_labels) >= torch.sum(predictions == higher_labels):
            gold_labels = lower_labels
            gold_labels[predictions == higher_labels] = higher_labels[predictions == higher_labels]
        else:
            gold_labels = higher_labels
            gold_labels[predictions == lower_labels] = lower_labels[predictions == lower_labels]
        return gold_labels

    def evaluation(self, outputs, test=False):
        pred_labels, gold_labels_low, gold_labels_high = [], [], []
        for output in outputs:
            gold_labels_low += output['gold_labels_low'].tolist()
            gold_labels_high += output['gold_labels_high'].tolist()
            pred_labels += output['pred_labels'].tolist()

        gold_labels_high = np.array(gold_labels_high)
        gold_labels_low = np.array(gold_labels_low)
        pred_labels = np.array(pred_labels)

        # pick higher or lower labels that the model performs better
        gold_labels = self.get_gold_labels(torch.from_numpy(pred_labels), torch.from_numpy(gold_labels_low),
                                           torch.from_numpy(gold_labels_high))
        gold_labels = gold_labels.numpy()

        eval_score = f1_score(gold_labels, pred_labels, average='macro')
        logs = {"score": eval_score}

        if test:
            eval_multiclass(self.logger.log_dir + '/sentence', gold_labels, pred_labels)
            with open(self.logger.log_dir + '/test_predictions.txt', 'w') as fw:
                fw.write('Sentence_Lv\n')
                for sent_lv in pred_labels:
                    fw.write('{0}\n'.format(sent_lv))
            
            with open(self.logger.log_dir + '/predictions.txt', 'w') as file:
                predictions_info = '\n'.join(['{} | {}'.format(str(pred[0]), str(target[0])) for pred, target in zip(pred_labels, gold_labels)])
                file.write(predictions_info)

        return logs

    def configure_optimizers(self):
        optimizer = transformers.AdamW(self.parameters(), lr=self.learning_rate)
        # Warm-up scheduler
        if self.warmup > 0:
            scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer

    def prepare_data(self):
        self.train_levels_a, self.train_levels_b, self.train_sents = read_corpus(
            self.corpus_path + '/train.tsv', self.num_labels, self.score_name)
        self.dev_levels_a, self.dev_levels_b, self.dev_sents = read_corpus(
            self.corpus_path + '/valid.tsv', self.num_labels, self.score_name)
        self.test_levels_a, self.test_levels_b, self.test_sents = read_corpus(
            self.test_corpus_path + '/test.tsv', self.num_labels, self.score_name)

    # return the dataloader for each split
    def train_dataloader(self):
        data_type = torch.float if self.num_labels == 1 else torch.long
        y_sent_a = torch.tensor(self.train_levels_a, dtype=data_type).unsqueeze(1)
        y_sent_b = torch.tensor(self.train_levels_b, dtype=data_type).unsqueeze(1)
        inputs = self.my_tokenize(self.train_sents)
        
        return DataLoader(CEFRDataset(inputs, y_sent_a, y_sent_b), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        data_type = torch.float if self.num_labels == 1 else torch.long
        y_sent_a = torch.tensor(self.dev_levels_a, dtype=data_type).unsqueeze(1)
        y_sent_b = torch.tensor(self.dev_levels_b, dtype=data_type).unsqueeze(1)
        inputs = self.my_tokenize(self.dev_sents)

        return DataLoader(CEFRDataset(inputs, y_sent_a, y_sent_b), batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        data_type = torch.float if self.num_labels == 1 else torch.long
        y_sent_a = torch.tensor(self.test_levels_a, dtype=data_type).unsqueeze(1)
        y_sent_b = torch.tensor(self.test_levels_b, dtype=data_type).unsqueeze(1)
        inputs = self.my_tokenize(self.test_sents)

        return DataLoader(CEFRDataset(inputs, y_sent_a, y_sent_b), batch_size=self.batch_size, shuffle=False)

    def my_tokenize(self, sents):
        max_seq_length = self.max_seq_length
        special_tokens_count = self.special_tokens_count
        
        for sent_idx, sent in enumerate(sents):
            tokens = sent
            #tokens = []
            #for i, word in enumerate(sent.split()):
            #    if len(tokens) >= max_seq_length - special_tokens_count:
            #        break
            #    word_pieces = self.tokenizer.tokenize(word)
            #    tokens.extend(word_pieces)
            #    
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[:(max_seq_length - special_tokens_count)]
            else:
                continue
            
            sents[sent_idx] = tokens
        
        inputs = self.tokenizer(sents, return_tensors="pt", padding=True,
                                is_split_into_words=True,
                                return_offsets_mapping=True)
        return inputs

    def retokenize_with_wordlvs(self, sents, wlvs):
        wlv_sequences = [[self.word_lv_dic[lv] for lv in wlv_list if lv >= 0 and lv < self.word_num_labels] for wlv_list
                         in
                         wlvs.clone().detach().numpy()]
        inputs = self.tokenizer(sents, text_pair=wlv_sequences, return_tensors="pt", padding=True,
                                is_split_into_words=True,
                                return_offsets_mapping=True)
        return inputs

    def wordlabel_to_tokenlabel(self, all_token_ids, all_offsets_mapping, labels):
        token_labels = torch.zeros_like(all_token_ids)
        for sid in range(all_token_ids.shape[0]):
            wid = -1
            for i, offset in enumerate(all_offsets_mapping[sid]):
                if offset[1] == 0:  # Special tokens like CLS, PAD # Faster but cannot handle self-added [SEP] token
                # if all_token_ids[sid][i] in self.tokenizer.all_special_ids: # Special tokens like CLS, PAD: Much slower
                    token_labels[sid, i] = -1
                    continue
                if offset[0] == 0:  # New word starts
                    wid += 1
                token_labels[sid, i] = labels[sid][wid]
        return token_labels
