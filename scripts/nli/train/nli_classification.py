import os
import torch
import sys

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

@register_task('nli_classification')
class NLIClassificationTask(LegacyFairseqTask):

    @staticmethod
    def add_args(parser):
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--max-positions', default=1024, type=int,
                            help='max input length')
        parser.add_argument('--model_type', default='nli', type=str,
                            help='nli|paraphrase')

    @classmethod
    def setup_task(cls, args, **kwargs):
        input_vocab = Dictionary.load(os.path.join(args.data, 'dict.input.txt'))
        label_vocab = Dictionary.load(os.path.join(args.data, 'dict.label.txt'))
        print('| [input] dictionary: {} types'.format(len(input_vocab)), file=sys.stderr)
        print('| [label] dictionary: {} types'.format(len(label_vocab)), file=sys.stderr)

        return NLIClassificationTask(args, input_vocab, label_vocab)

    def __init__(self, args, input_vocab, label_vocab):
        super().__init__(args)
        self.input_vocab = input_vocab
        self.label_vocab = label_vocab

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        if split in ["train", "valid", "test"]:
            prefix = os.path.join(self.args.data, split)
            premise_path = prefix + '.premise'
            hypothesis_path = prefix + '.hypothesis'
            label_path = prefix + '.label'
        else:
            premise_path, hypothesis_path, label_path = split

        # Read input premises.
        premises, premises_lengths = [], []
        with open(premise_path, encoding='utf-8') as file:
            for line in file:
                sentence = line.strip()

                # Tokenize the sentence, splitting on spaces
                tokens = self.input_vocab.encode_line(
                    "<s> " + sentence, add_if_not_exist=False,
                )

                premises.append(tokens)
                premises_lengths.append(tokens.numel())

        # Read input hypotheses.
        hypotheses, hypotheses_lengths = [], []
        with open(hypothesis_path, encoding='utf-8') as file:
            for line in file:
                sentence = line.strip()

                # Tokenize the sentence, splitting on spaces
                tokens = self.input_vocab.encode_line(
                    "<s> " + sentence, add_if_not_exist=False,
                )

                hypotheses.append(tokens)
                hypotheses_lengths.append(tokens.numel())

        # Read labels.
        labels = []
        with open(label_path, encoding='utf-8') as file:
            for line in file:
                label = line.strip()
                labels.append(
                    # Convert label to a numeric ID.
                    torch.LongTensor([int(label)])
                )

        assert len(premises) == len(labels), f"{len(premises)} != {len(labels)}"
        print('| {} {} {} premise examples'.format(self.args.data, split, len(premises)))

        assert len(hypotheses) == len(labels), f"{len(hypotheses)} != {len(labels)}"
        print('| {} {} {} hypothesis examples'.format(self.args.data, split, len(hypotheses)))

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "premises_tokens":MonolingualDataset(
                    dataset=premises,
                    sizes=premises_lengths,
                    src_vocab=self.input_vocab,
                    shuffle=False
                ),
                "hypotheses_tokens": MonolingualDataset(
                    dataset=hypotheses,
                    sizes=hypotheses_lengths,
                    src_vocab=self.input_vocab,
                    shuffle=False
                ),
            },
            "target": RawLabelDataset(labels),
            "premises_ntokens": NumelDataset(premises, reduce=True),
            "hypotheses_ntokens": NumelDataset(hypotheses, reduce=True),
            "ntokens": NumelDataset(premises, reduce=True),
            "nsentences": NumSamplesDataset(),
        }

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[
                dataset["net_input"]["premises_tokens"].sizes,
                dataset["net_input"]["hypotheses_tokens"].sizes
            ],
        )

        self.datasets[split] = nested_dataset
        return self.datasets[split]

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and the "target"
        # has max length 1.
        return self.args.max_positions, self.args.max_positions, 1

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.input_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.label_vocab

    def build_generator(self, models, args, **unused):
        from fairseq.nli_generator import NLIGenerator

        return NLIGenerator(
            models[0]
        )