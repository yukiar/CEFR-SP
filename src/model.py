import torch, random, itertools, tqdm
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from util import mean_pooling, read_corpus, CEFRDataset, convert_numeral_to_six_levels
from model_base import LevelEstimaterBase


class LevelEstimaterClassification(LevelEstimaterBase):
    def __init__(self, corpus_path, test_corpus_path, pretrained_model, problem_type, with_ib, with_loss_weight,
                 attach_wlv, num_labels,
                 word_num_labels, alpha,
                 ib_beta,
                 batch_size,
                 learning_rate,
                 warmup,
                 lm_layer):
        super().__init__(corpus_path, test_corpus_path, pretrained_model, with_ib, attach_wlv, num_labels,
                         word_num_labels, alpha,
                         batch_size,
                         learning_rate, warmup, lm_layer)
        self.save_hyperparameters()

        self.problem_type = problem_type
        self.with_loss_weight = with_loss_weight
        self.ib_beta = ib_beta
        self.dropout = nn.Dropout(0.1)

        if self.problem_type == "regression":
            self.slv_classifier = nn.Linear(self.lm.config.hidden_size, 1)
            self.loss_fct = nn.MSELoss()
        else:
            self.slv_classifier = nn.Linear(self.lm.config.hidden_size, self.CEFR_lvs)
            if self.with_loss_weight:
                train_sentlv_weights = self.precompute_loss_weights()
                self.loss_fct = nn.CrossEntropyLoss(weight=train_sentlv_weights)
            else:
                self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, inputs):
        # in lightning, forward defines the prediction/inference actions
        outputs, information_loss = self.encode(inputs)
        outputs = mean_pooling(outputs, attention_mask=inputs['attention_mask'])
        logits = self.slv_classifier(self.dropout(outputs))

        if self.problem_type == "regression":
            predictions = convert_numeral_to_six_levels(logits.detach().clone().cpu().numpy())
        else:
            predictions = torch.argmax(torch.softmax(logits.detach().clone(), dim=1), dim=1, keepdim=True)

        loss = None
        if 'slabels_high' in inputs:
            if self.problem_type == "regression":
                labels = (inputs['slabels_high'] + inputs['slabels_low']) / 2
                cls_loss = self.loss_fct(logits.squeeze(), labels.squeeze())
            else:
                labels = self.get_gold_labels(predictions, inputs['slabels_low'].detach().clone(),
                                              inputs['slabels_high'].detach().clone())
                cls_loss = self.loss_fct(logits.view(-1, self.CEFR_lvs), labels.view(-1))

            loss = cls_loss
            logs = {"loss": cls_loss}

        predictions = predictions.cpu().numpy()

        return (loss, predictions, logs) if loss is not None else predictions

    def step(self, batch):
        loss, predictions, logs = self.forward(batch)
        return loss, logs

    def _shared_eval_step(self, batch):
        loss, predictions, logs = self.forward(batch)

        gold_labels_low = batch['slabels_low'].cpu().detach().clone().numpy()
        gold_labels_high = batch['slabels_high'].cpu().detach().clone().numpy()
        golds_predictions = {'gold_labels_low': gold_labels_low, 'gold_labels_high': gold_labels_high,
                             'pred_labels': predictions}

        return logs, golds_predictions

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        self.log_dict({f"train_{k}": v for k, v in logs.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        logs, golds_predictions = self._shared_eval_step(batch)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return golds_predictions

    def validation_epoch_end(self, outputs):
        logs = self.evaluation(outputs)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})

    def test_step(self, batch, batch_idx):
        logs, golds_predictions = self._shared_eval_step(batch)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        return golds_predictions

    def test_epoch_end(self, outputs):
        logs = self.evaluation(outputs, test=True)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})


class LevelEstimaterContrastive(LevelEstimaterBase):
    def __init__(self, corpus_path, test_corpus_path, pretrained_model, problem_type, with_ib, with_loss_weight,
                 attach_wlv, num_labels,
                 word_num_labels,
                 num_prototypes,
                 alpha,
                 ib_beta,
                 batch_size,
                 learning_rate,
                 warmup,
                 lm_layer):
        super().__init__(corpus_path, test_corpus_path, pretrained_model, with_ib, attach_wlv, num_labels,
                         word_num_labels, alpha,
                         batch_size,
                         learning_rate, warmup, lm_layer)
        self.save_hyperparameters()

        self.problem_type = problem_type
        self.num_prototypes = num_prototypes
        self.with_loss_weight = with_loss_weight
        self.ib_beta = ib_beta

        self.prototype = nn.Embedding(self.CEFR_lvs * self.num_prototypes, self.lm.config.hidden_size)
        # nn.init.xavier_normal_(self.prototype.weight)  # Xavier initialization
        # nn.init.orthogonal_(self.prototype.weight)  # Make prototype vectors orthogonal

        if self.with_loss_weight:
            loss_weights = self.precompute_loss_weights()
            self.loss_fct = nn.CrossEntropyLoss(weight=loss_weights)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        outputs, information_loss = self.encode(batch)
        outputs = mean_pooling(outputs, attention_mask=batch['attention_mask'])

        # positive: compute cosine similarity
        outputs = torch.nn.functional.normalize(outputs)
        positive_prototypes = torch.nn.functional.normalize(self.prototype.weight)
        logits = torch.mm(outputs, positive_prototypes.T)
        logits = logits.reshape((-1, self.num_prototypes, self.CEFR_lvs))
        logits = logits.mean(dim=1)

        # prediction
        predictions = torch.argmax(torch.softmax(logits.detach().clone(), dim=1), dim=1, keepdim=True)

        loss = None
        if 'slabels_high' in batch:
            labels = self.get_gold_labels(predictions, batch['slabels_low'].detach().clone(),
                                          batch['slabels_high'].detach().clone())
            # cross-entropy loss
            cls_loss = self.loss_fct(logits.view(-1, self.CEFR_lvs), labels.view(-1))

            loss = cls_loss
            logs = {"loss": loss}

        predictions = predictions.cpu().numpy()

        return (loss, predictions, logs) if loss is not None else predictions

    def _shared_eval_step(self, batch):
        loss, predictions, logs = self.forward(batch)

        gold_labels_low = batch['slabels_low'].cpu().detach().clone().numpy()
        gold_labels_high = batch['slabels_high'].cpu().detach().clone().numpy()
        golds_predictions = {'gold_labels_low': gold_labels_low, 'gold_labels_high': gold_labels_high,
                             'pred_labels': predictions}

        return logs, golds_predictions

    def on_train_start(self) -> None:
        # Init with BERT embeddings
        epcilon = 1.0e-6
        higher_labels, lower_labels = [], []
        prototype_initials = torch.full((self.CEFR_lvs, self.lm.config.hidden_size), fill_value=epcilon).to(self.device)

        self.lm.eval()
        for batch in tqdm.tqdm(self.train_dataloader(), leave=False, desc='init prototypes'):
            higher_labels += batch['slabels_high'].squeeze().detach().clone().numpy().tolist()
            lower_labels += batch['slabels_low'].squeeze().detach().clone().numpy().tolist()
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.lm(batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
                outputs_mean = mean_pooling(outputs.hidden_states[self.lm_layer],
                                            attention_mask=batch['attention_mask'])
            for lv in range(self.CEFR_lvs):
                prototype_initials[lv] += outputs_mean[
                    (batch['slabels_low'].squeeze() == lv) | (batch['slabels_high'].squeeze() == lv)].sum(0)
        if not self.with_ib:
            self.lm.train()

        higher_labels = torch.tensor(higher_labels)
        lower_labels = torch.tensor(lower_labels)
        for lv in range(self.CEFR_lvs):
            denom = torch.count_nonzero((higher_labels == lv) | (lower_labels == lv)) + epcilon
            prototype_initials[lv] = prototype_initials[lv] / denom

        var = torch.var(prototype_initials).item() * 0.05 # Add Gaussian noize with 5% variance of the original tensor
        # prototype_initials = torch.repeat_interleave(prototype_initials, self.num_prototypes, dim=0)
        prototype_initials = prototype_initials.repeat(self.num_prototypes, 1)
        noise = (var ** 0.5) * torch.randn(prototype_initials.size()).to(self.device)
        prototype_initials = prototype_initials + noise  # Add Gaussian noize
        self.prototype.weight = nn.Parameter(prototype_initials)
        nn.init.orthogonal_(self.prototype.weight)  # Make prototype vectors orthogonal

        # # Init with Xavier
        # nn.init.xavier_normal_(self.prototype.weight)  # Xavier initialization

    def training_step(self, batch, batch_idx):
        loss, predictions, logs = self.forward(batch)
        self.log_dict({f"train_{k}": v for k, v in logs.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        logs, golds_predictions = self._shared_eval_step(batch)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return golds_predictions

    def validation_epoch_end(self, outputs):
        logs = self.evaluation(outputs)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})

    def test_step(self, batch, batch_idx):
        logs, golds_predictions = self._shared_eval_step(batch)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        return golds_predictions

    def test_epoch_end(self, outputs):
        logs = self.evaluation(outputs, test=True)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
