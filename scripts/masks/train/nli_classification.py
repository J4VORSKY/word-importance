import torch

from fairseq.models import BaseFairseqModel, register_model
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import register_model_architecture
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerConfig,
    TransformerModelBase
)

from fairseq.models.nli_classifier import BiEncoderClassifier

import re
import torch
import pytorch_lightning as pl
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertConfig,
    get_constant_schedule_with_warmup,
    get_constant_schedule,
)
from ..utils.util import accuracy_precision_recall_f1

import os
import torch

from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.data import (
    MonolingualDataset,
    IdDataset,
    NumelDataset,
    NumSamplesDataset,
    RawLabelDataset,
    NestedDictionaryDataset,
    Dictionary,
)


def load_sst(path, tokenizer, lower=False):
    premises, hypotheses, labels = [], [], []
    premises_lengths, hypotheses_lengths = [], []

    with open(path + '.premise', encoding='utf-8') as file:
        for line in file:
            sentence = " ".join(line.strip().split()[:120])
            tokens = tokenizer.encode_line(sentence, add_if_not_exist=False)
            premises_lengths.append(tokens.numel())

            tokens = torch.cat([tokens, torch.ones(128 - len(tokens))]).tolist()
            premises.append(tokens)

    with open(path + '.hypothesis', encoding='utf-8') as file:
        for line in file:
            sentence = " ".join(line.strip().split()[:120])
            tokens = tokenizer.encode_line(sentence, add_if_not_exist=False)
            hypotheses_lengths.append(tokens.numel())

            tokens = torch.cat([tokens, torch.ones(128 - len(tokens))]).tolist()
            hypotheses.append(tokens)

    with open(path + '.label', encoding='utf-8') as file:
        for line in file:
            label = line.strip()
            tokens = torch.LongTensor([int(label)])

            labels.append(tokens)

    tensor_dataset = (
        torch.tensor(premises, dtype=torch.long),
        torch.tensor(premises_lengths, dtype=torch.long),
        torch.tensor(hypotheses, dtype=torch.long),
        torch.tensor(hypotheses_lengths, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )

    return torch.utils.data.TensorDataset(*tensor_dataset)


class SentimentClassificationSST(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.tokenizer = Dictionary.load(hparams.dictionary)

    def prepare_data(self):
        # assign to use in dataloaders
        if not hasattr(self, "train_dataset"):
            self.train_dataset = load_sst(
                self.hparams.train_filename, self.tokenizer,
            )
        if not hasattr(self, "val_dataset"):
            self.val_dataset = load_sst(
                self.hparams.val_filename, self.tokenizer,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size
        )

    def training_step(self, batch, batch_idx=None):
        _, _, _, _, labels = batch

        logits = self.forward(**batch)[0]

        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none").mean(
            -1
        )

        acc, _, _, f1 = accuracy_precision_recall_f1(
            logits.argmax(-1), labels, average=True
        )

        outputs_dict = {
            "acc": acc,
            "f1": f1,
        }

        outputs_dict = {
            "loss": loss,
            **outputs_dict,
            "log": outputs_dict,
            "progress_bar": outputs_dict,
        }

        outputs_dict = {
            "{}{}".format("" if self.training else "val_", k): v
            for k, v in outputs_dict.items()
        }

        return outputs_dict

    def validation_step(self, batch, batch_idx=None):
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):

        outputs_dict = {
            k: sum(e[k] for e in outputs) / len(outputs) for k in ("val_acc", "val_f1")
        }

        outputs_dict = {
            "val_loss": -outputs_dict["val_f1"],
            **outputs_dict,
            "log": outputs_dict,
        }

        return outputs_dict

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(self.parameters(), self.hparams.learning_rate),
        ]
        schedulers = [
            {
                "scheduler": get_constant_schedule_with_warmup(optimizers[0], 200),
                "interval": "step",
            },
        ]

        return optimizers, schedulers


class BertSentimentClassificationSST(SentimentClassificationSST):
    def __init__(self, hparams):
        super().__init__(hparams)

        data_path = "../../" + hparams.dictionary[:-len("dict.input.txt")]

        self.net = BiEncoderClassifier.from_pretrained(
            hparams.model, checkpoint_file="checkpoint6.pt",
            data_name_or_path=data_path
        )

        self.net = self.net.models[0].classifier

        self.net.config = hparams
        self.net.config.num_hidden_layers = 6
        self.net.config.hidden_size = 512

    def forward(self, premises, premise_lengths, hypotheses, hypothesis_lengths):
        return self.net(premises, premise_lengths, hypotheses, hypothesis_lengths)