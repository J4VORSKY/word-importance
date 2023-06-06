import torch

from fairseq.models import BaseFairseqModel, register_model
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import register_model_architecture
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerConfig,
    TransformerModelBase
)


class BiEncoderClassifier(BaseFairseqModel):

    def __init__(self, args, vocabulary):
        super(BiEncoderClassifier, self).__init__()

        self.args = args

        self.embed_tokens = TransformerModelBase.build_embedding(
            args, vocabulary, args.encoder_embed_dim
        )

        self.enc1 = TransformerEncoder(args, vocabulary, self.embed_tokens)

        if args.model_type == "nli":
            self.linear = torch.nn.Linear(3 * args.encoder_embed_dim, 3)
        elif args.model_type == "nli-max-pool":
            self.linear = torch.nn.Linear(3 * args.encoder_embed_dim, 3)
        elif args.model_type == "paraphrase":
            self.linear = torch.nn.Linear(2 * args.encoder_embed_dim, 2)
        elif args.model_type == "paraphrase-diff":
            self.linear = torch.nn.Linear(3 * args.encoder_embed_dim, 2)
        else:
            raise Exception(f"Model type {args.model_type} is not supported. "
                "Choose from: nli, paraphrase.")
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, enc1_tokens, enc1_lengths, enc2_tokens, enc2_lengths):
        if self.args.model_type == "nli":
            return self.__forward_nli(
                enc1_tokens, enc1_lengths, enc2_tokens, enc2_lengths)
        elif self.args.model_type == "nli-max-pool":
            return self.__forward_nli_max_pool(
                enc1_tokens, enc1_lengths, enc2_tokens, enc2_lengths)
        elif self.args.model_type == "paraphrase":
            return self.__forward_paraphrase(
                enc1_tokens, enc1_lengths, enc2_tokens, enc2_lengths)
        elif self.args.model_type == "paraphrase-diff":
            return self.__forward_nli(
                enc1_tokens, enc1_lengths, enc2_tokens, enc2_lengths)

    def __forward_paraphrase(self, enc1_tokens, enc1_lengths, enc2_tokens, enc2_lengths):
        enc1_output = self.enc1(enc1_tokens, enc1_lengths)["encoder_out"][0]
        enc2_output = self.enc1(enc2_tokens, enc2_lengths)["encoder_out"][0]

        max_enc1 = enc1_output[0, :, :].view(-1, self.args.encoder_embed_dim)
        max_enc2 = enc2_output[0, :, :].view(-1, self.args.encoder_embed_dim)

        concat = torch.cat((max_enc1, max_enc2), dim=1)

        logits = self.linear(concat)

        return logits
    
    def __forward_nli(self, enc1_tokens, enc1_lengths, enc2_tokens, enc2_lengths):
        enc1_output = self.enc1(enc1_tokens, enc1_lengths)["encoder_out"][0]
        enc2_output = self.enc1(enc2_tokens, enc2_lengths)["encoder_out"][0]

        mean_enc1 = torch.mean(enc1_output, dim=0)
        mean_enc2 = torch.mean(enc2_output, dim=0)

        concat = torch.cat((
            mean_enc1,
            mean_enc2,
            torch.abs(torch.sub(mean_enc1, mean_enc2))
        ), dim=1)

        logits = self.linear(concat)

        return logits

    def __forward_nli_max_pool(self, enc1_tokens, enc1_lengths, enc2_tokens, enc2_lengths):
        enc1_output = self.enc1(enc1_tokens, enc1_lengths)["encoder_out"][0]
        enc2_output = self.enc1(enc2_tokens, enc2_lengths)["encoder_out"][0]

        max_enc1 = enc1_output[0, :, :].view(-1, self.args.encoder_embed_dim)
        max_enc2 = enc2_output[0, :, :].view(-1, self.args.encoder_embed_dim)

        concat = torch.cat((
            max_enc1,
            max_enc2,
            torch.abs(torch.sub(max_enc1, max_enc2))
        ), dim=1)

        logits = self.linear(concat)

        return logits
    
@register_model('nli_classifier')
class NLIClassifier(BaseFairseqModel):

    @staticmethod
    def add_args(parser):
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task):
        classifier = BiEncoderClassifier(args, task.input_vocab)

        return NLIClassifier(classifier, task.input_vocab)

    def __init__(self, classifier, vocabulary):
        super(NLIClassifier, self).__init__()

        self.classifier = classifier
        self.vocabulary = vocabulary

    def forward(self, premises_tokens, hypotheses_tokens):
        return self.classifier(
            premises_tokens["net_input"]["src_tokens"],
            premises_tokens["net_input"]["src_lengths"],
            hypotheses_tokens["net_input"]["src_tokens"],
            hypotheses_tokens["net_input"]["src_lengths"]
        )

@register_model_architecture('nli_classifier', 'nli_bi_encoder')
def nli_bi_encoder(args):
    pass