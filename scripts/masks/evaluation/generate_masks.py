import argparse
import os
import sys

import torch
from diffmask.models.nli_classification_diffmask import BertSentimentClassificationSSTDiffMask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reads lines and outputs: LINE <tab> LIST OF ATTRIBUTES. \
        Lists are separated by <tab>, values within one list by <space>."
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gate_bias", action="store_true")
    parser.add_argument("--seed", type=float, default=0)
    parser.add_argument("--float_precision", type=int, default=8)
    parser.add_argument("--input_file", type=str, default=None) # "msr_masks.txt"
    parser.add_argument("--output_file", type=str, default=None) # "../data/msr/compression.tok"
    parser.add_argument(
        "--model_path",
        type=str,
        default="./outputs/6-l.8-h.1024-ffn.512-emb.tok/epoch=4-val_acc=0.87-val_f1=0.86-val_l0=0.31.ckpt",
    )

    args, _ = parser.parse_known_args()

    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = "cpu"

    model = BertSentimentClassificationSSTDiffMask.load_from_checkpoint(args.model_path).to(device)
    model.freeze()

    lines, lines_lengths, encoded_lines, encoded_lines_lengths, attributions = [], [], [], [], []

    def get_attributions():
        global encoded_lines, encoded_lines_lengths

        enc = torch.tensor(encoded_lines, device=device, dtype=torch.long)
        len = torch.tensor(encoded_lines_lengths, device=device, dtype=torch.long)

        encoded_lines, encoded_lines_lengths = [], []

        return model.forward_explainer(enc, len, enc, len, attribution=True).exp().cpu()

    input_stream = sys.stdin if args.input_file == None else open(args.input_file, "r", encoding="utf8")
    for i, line in enumerate(input_stream):
        lines.append(line.strip())

        # converting tokens to numbers based on the dictionary
        encoded = model.tokenizer.encode_line(line, add_if_not_exist=False)[:128].to(device)

        lines_lengths.append(encoded.numel())
        encoded_lines_lengths.append(encoded.numel())

        # aligning all the inputs to 128 tokens
        tokens = torch.cat([encoded, torch.ones(128 - len(encoded))]).tolist()
        encoded_lines.append(tokens)

        if (i + 1) % args.batch_size == 0:
            attributions += get_attributions()

    attributions += get_attributions()
    input_stream.close()

    output_stream = sys.stdout if args.output_file == None else open(args.output_file, "w", encoding="utf8")
    for line, length, attr in zip(lines, lines_lengths, attributions):
        values = "\t".join([" ".join([('{0:.' + str(args.float_precision) + 'f}').format(x)
            for x in v.tolist()]) for v in attr[:length - 1]])

        print(f"{line}\t{values}", file=output_stream)
    
    output_stream.close()
