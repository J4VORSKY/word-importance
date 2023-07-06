from operator import index
import random
import torch
import sys
import argparse

def filter_sentence(sentence, masks, threshold, mode, filtering):
    """
    Returns list of words which left after filtering.
    """
    def compare(s, t):
        if filtering == "low-scored":
            return s >= t
        elif filtering == "high-scored":
            return s <= t
        else:
            raise f"Filtering `{filtering}` is not recognized. Use --filtering (low-scored|high-scored) instead."

    masks = torch.tensor(masks)
    for _ in range(masks.size(0)):
        if mode == "mask":
            return ([word for word, score in zip(sentence.split(), masks) if score >= threshold],
                " ".join([word if score >= threshold else "[" + word + "]"
                    for word, score in zip(sentence.split(), masks)]))
        elif mode == "length":
            if filtering == "low-scored":
                threshold = sorted(masks)[int(len(masks) * threshold)]
            elif filtering == "high-scored":
                threshold = sorted(masks, reverse=True)[int(len(masks) * threshold)]

            return ([word for word, score in zip(sentence.split(), masks) if compare(score, threshold)],
                " ".join([word if compare(score, threshold) else "[" + word + "]" # "\x1b[31m" + word + "\x1b[0m"
                    for word, score in zip(sentence.split(), masks)]))
        elif mode == "random":
            if filtering == "low-scored":
                threshold = sorted(masks)[int(len(masks) * threshold)]
            elif filtering == "high-scored":
                threshold = sorted(masks, reverse=True)[int(len(masks) * threshold)]

            mask_count = len([score for score in masks if compare(score, threshold)])
            indices = random.sample(list(range(len(masks))), mask_count)
            return ([word for index, word in enumerate(sentence.split()) if index in indices],
                " ".join([word if index in indices else "[" + word + "]"
                    for index, word in enumerate(sentence.split())]))
        else:
            raise f"Filtering mode `{mode}` is not recognized. Use --mode (length|mask|random) instead."
                
def parse_line(line):
    items = line.split("\t")
    return items[0], [float(value) for value in items[1].split()]

if __name__ == "__main__":
    # Expects one value per token. It is needed to filter the output from generate_masks.py, reducing to just one value.
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--mode", type=str, default="length", help="length|mask|random")
    parser.add_argument("--filtering", type=str, default="low-scored", help="low-scored|high-scored")

    args, _ = parser.parse_known_args()

    input_stream = sys.stdin if args.input_file == None else open(args.input_file, "r", encoding="utf8")
    output_stream = sys.stdout if args.output_file == None else open(args.output_file, "w", encoding="utf8")
    for line in input_stream:
        if line.strip() == "":
            print(file=output_stream)
            continue

        sentence, masks = parse_line(line)
        filtered, to_display = filter_sentence(sentence, masks, args.threshold, mode=args.mode, filtering=args.filtering)
        print(f"{sentence}\t{' '.join(filtered)}\t{to_display}", file=output_stream)

    output_stream.close()