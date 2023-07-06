import argparse
import sys
import torch

def parse_line(line):
    items = line.split("\t")
    return items[0], [[float(v) for v in mask.split()] for mask in items[1:]]

def aggregate_masks(masks, mode, scale):
    masks = torch.tensor(masks)
    for _ in range(masks.size(0)):
        to_plot = (masks / masks.abs().max(0, keepdim=True).values)

        if mode == "last":
            min_val = to_plot[..., -1].min()
            last_scaled = ((to_plot[..., -1] - min_val) / (to_plot[..., -1] - min_val).max()).tolist()

            to_return = to_plot[..., -1] if not scale else last_scaled

            return " ".join([('{0:.' + str(args.float_precision) + 'f}').format(x) for x in to_return])
        elif mode == "average":
            average = torch.mean(to_plot, 1)
            average_min = average.min()
            average_scaled = ((average - average_min) / (average - average_min).max()).tolist()

            to_return = average if not scale else average_scaled
            
            return " ".join([('{0:.' + str(args.float_precision) + 'f}').format(x) for x in to_return])
        else:
            raise "Aggregating mode `{mode}` is not recognized. Use --mode {last,average} instead."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--mode", type=str, default="last", help="last|average")
    parser.add_argument("--scale", default=False, action="store_true", help="whether to scale the scores to [0, 1]")
    parser.add_argument("--float_precision", type=int, default=8)

    args, _ = parser.parse_known_args()

    input_stream = sys.stdin if args.input_file == None else open(args.input_file, "r", encoding="utf8")
    output_stream = sys.stdout if args.output_file == None else open(args.output_file, "w", encoding="utf8")
    
    for line in input_stream:
        if line.strip() == "":
            print(file=output_stream)
            continue

        sentence, masks = parse_line(line)
        aggregated = aggregate_masks(masks, mode=args.mode, scale=args.scale)
        print(f"{sentence}\t{aggregated}", file=output_stream)

    input_stream.close()
    output_stream.close()
