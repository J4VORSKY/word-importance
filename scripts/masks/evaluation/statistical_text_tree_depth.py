import argparse
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--plot_path", type=str, default=None)

    args, _ = parser.parse_known_args()

    input_stream = sys.stdin if args.input_file == None else open(args.input_file, "r", encoding="utf8")
    output_stream = sys.stdout if args.output_file == None else open(args.output_file, "w", encoding="utf8")

    depths_pearson = []
    scores_pearson = []

    depths_pearson_up_N = []
    scores_pearson_up_N = []

    depth_to_mask = defaultdict(list)

    all_pos = []
    all_scores = []

    pos_to_mask = defaultdict(list)
    for line in input_stream:
        items = line.strip().split("\t")
        if line.strip() == "": continue

        orig = items[0]
        masks = [float(val) for val in items[1].split()]
        ids = [int(val) for val in items[2].split()]
        lemmas = items[3].split()
        pos = items[4].split()
        heads = [int(val) for val in items[5].split()]


        for p, m in zip(pos, masks):
            pos_to_mask[p].append(m)
            all_pos.append(p)
            all_scores.append(m)

        depths = [1] * len(ids)
        while sum(heads) != 0:
            for index, head in enumerate(heads):
                if head == 0: continue
                if heads[head - 1] == 0:
                    depths[index] = 1 + depths[head - 1]
                    heads[index] = 0

        # print(f"{orig}\t{' '.join([str(m) for m in masks])}\t{' '.join([str(d) for d in depths])}", file=output_stream)
        # if (6 in depths) and (masks[depths.index(6)] > 0.7):
        #     print(f"{orig}\t{' '.join([f'{m:.2f}' for m in masks])}\t{' '.join([str(d) for d in depths])}", file=output_stream)

        for d, m in zip(depths[:-1], masks[:-1]):
            depth_to_mask[d].append(m)
            depths_pearson.append(d)
            scores_pearson.append(m)

            if d <= 5:
                depths_pearson_up_N.append(d)
                scores_pearson_up_N.append(m)

    print("depth, avg, var, std, #")
    for d in sorted(depth_to_mask.keys()):
        print(d,
            f"{np.mean(depth_to_mask[d]):.2f}",
            f"{np.var(depth_to_mask[d]):.2f}",
            f"{np.std(depth_to_mask[d]):.2f}",
            f"{len(depth_to_mask[d])}",
        sep="    ")

    print()
    print("\t" + "\t".join(map(str, sorted(depth_to_mask.keys()))))
    for d1 in sorted(depth_to_mask.keys()):
        print(d1, end="\t")
        for d2 in sorted(depth_to_mask.keys()):
            if d1 > d2:
                print(f"{scipy.stats.ttest_ind(depth_to_mask[d1], depth_to_mask[d2]).pvalue:.3f}", end="\t")
            # print(f"({d1},{d2})", scipy.stats.ttest_ind(depth_to_mask[d1], depth_to_mask[d2]).pvalue)
        print()

    print()
    print("Pearson correlation:", scipy.stats.pearsonr(depths_pearson, scores_pearson))
    print("Spearman correlation:", scipy.stats.spearmanr(depths_pearson, scores_pearson))

    print()
    print("POS, avg, var, std, #")
    for d in sorted(pos_to_mask.keys()):
        print(d,
            f"{np.mean(pos_to_mask[d]):.2f}",
            f"{np.var(pos_to_mask[d]):.2f}",
            f"{np.std(pos_to_mask[d]):.2f}",
            f"{len(pos_to_mask[d])}",
        sep="    ")

    
    fig = plt.figure(figsize=(3, 6))
    
    data = pd.DataFrame({"POS": all_pos, "Scores": all_scores})
    
    sns.barplot(data=data, y="POS", x="Scores", orient="h", estimator=np.mean,
        order=data.groupby('POS').Scores.agg('mean').sort_values(ascending=False).index, color='steelblue')
    
    plt.tight_layout()

    plt.savefig(args.plot_path, bbox_inches="tight", pad_inches=0)
    
    
    # print()
    # print("If only up to level 6")
    # print("Pearson correlation:", scipy.stats.pearsonr(depths_pearson_up_N, scores_pearson_up_N))
    # print("Spearman correlation:", scipy.stats.spearmanr(depths_pearson_up_N, scores_pearson_up_N))

    # depth_to_mask = {key: depth_to_mask[key] for key in sorted(depth_to_mask.keys())}

    # fig, ax = plt.subplots(1, 7, figsize=(10, 5))
    # i = 0
    # for depth, masks in depth_to_mask.items():
    #     sns.distplot(masks, bins=[x / 100 for x in range(0, 105, 5)], ax=ax[i], kde=False, label=f"n = {len(masks)}")
    #     ax[i].title.set_text(f"Depth {depth}")
    #     ax[i].set_ylim([0, 270])
    #     ax[i].set_xlim([0, 1])
    #     i += 1
    # # fig.legend(depth_to_mask.keys())
    # plt.tight_layout()
    # plt.savefig("./syntactic_tree.png")