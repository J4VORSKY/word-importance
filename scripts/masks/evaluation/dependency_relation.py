import collections
import argparse
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--json_path", type=str, default=None)

    args, _ = parser.parse_known_args()

    input_stream = sys.stdin if args.input_file == None else open(args.input_file, "r", encoding="utf8")
    output_stream = sys.stdout if args.output_file == None else open(args.output_file, "w", encoding="utf8")

    depth_to_mask = defaultdict(list)
    depth_to_mask_upto5 = defaultdict(list)
    depth_to_mask_reset = defaultdict(list)
    deprel_to_mask = defaultdict(list)
    aux_counter = collections.Counter()
    aux_count = 0

    for line in input_stream:
        if line.strip() == "": continue
        items = line.strip().split("\t")

        orig = items[0]
        masks = [float(val) for val in items[1].split()]
        ids = [int(val) for val in items[2].split()]
        lemmas = items[3].split()
        pos = items[4].split()
        heads_orig = [int(val) for val in items[5].split()]
        deprels = [d.split(":")[0] if ":" in d else d for d in items[6].split()]
        heads = [h for h in heads_orig]

        # depth analysis
        depths = [1] * len(ids)
        while sum(heads) != 0:
            for index, head in enumerate(heads):
                if head == 0: continue
                if heads[head - 1] == 0:
                    depths[index] = 1 + depths[head - 1]
                    heads[index] = 0

        for d, m, p in zip(depths, masks, pos):
            if p == "PUNCT": continue
            depth_to_mask[d].append(m)
        
        if all(d <= 1 for d in depths):
            for d, m, p in zip(depths, masks, pos):
                if p == "PUNCT": continue
                depth_to_mask_upto5[d].append(m)

        # depth with reset
        depths_reset = [1] * len(ids)
        heads = [h for h in heads_orig]
        while sum(heads) != 0:
            for index, head in enumerate(heads):
                if head == 0: continue
                if heads[head - 1] == 0:
                    # if lemmas[index] == "people" and (masks[index] > 0.9999):
                    #     print(line)
                    if lemmas[index] == "people" and (masks[index] > 0.8) and (masks[index] < 0.9):
                        # aux_counter.update([orig.split()[index]])
                        # aux_count += 1
                        print(line)
                    if deprels[index] in ["acl", "advcl", "csubj", "ccomp", "xcomp"]:
                        depths_reset[index] = 1
                    else:
                        depths_reset[index] = 1 + depths_reset[head - 1]
                    #if depths_reset[index] == 8: print(line)
                    heads[index] = 0

        for d, m, p in zip(depths_reset, masks, pos):
            if p == "PUNCT": continue
            depth_to_mask_reset[d].append(m)

        # print(f"{orig}\t{' '.join([str(m) for m in masks])}\t{' '.join([str(d) for d in depths])}", file=output_stream)
        # if (6 in depths) and (masks[depths.index(6)] > 0.7):
        #     print(f"{orig}\t{' '.join([f'{m:.2f}' for m in masks])}\t{' '.join([str(d) for d in depths])}", file=output_stream)

        # dependence analysis
        for i, d in enumerate(deprels):
            if d != "root":
                # if d == "expl": print(masks[i], masks[heads_orig[i] - 1])
                deprel_to_mask[d].append(masks[i] - masks[heads_orig[i] - 1])
            else:
                deprel_to_mask[d].append(masks[i])

    print(r"\bf Depth & \bf Avg & \bf Std & Tokens \\\hline")
    for d in sorted(depth_to_mask.keys()):
        print(d,
            f"{np.mean(depth_to_mask[d]):.2f}",
            # f"{np.var(depth_to_mask[d]):.2f}",
            f"{np.std(depth_to_mask[d]):.2f}",
            f"{len(depth_to_mask[d])}",
        sep=" & ", end=" \\\\\n")
    all_values = []
    for d in depth_to_mask.values():
        for v in d:
            all_values.append(v)
    print(f"All & {np.mean(all_values):.2f} & {np.std(all_values):.2f} & {len(all_values)} \\\\")
    
    print("depth, avg, var, std, #")
    for d in sorted(depth_to_mask_reset.keys()):
        print(d,
            f"{np.mean(depth_to_mask_reset[d]):.2f}",
            f"{np.var(depth_to_mask_reset[d]):.2f}",
            f"{np.std(depth_to_mask_reset[d]):.2f}",
            f"{len(depth_to_mask_reset[d])}",
        sep="    ")

    print(aux_count, aux_counter)

    # print("depth, avg, var, std, #")
    # for d in sorted(depth_to_mask_upto5.keys()):
    #     print(d,
    #         f"{np.mean(depth_to_mask_upto5[d]):.2f}",
    #         f"{np.var(depth_to_mask_upto5[d]):.2f}",
    #         f"{np.std(depth_to_mask_upto5[d]):.2f}",
    #         f"{len(depth_to_mask_upto5[d])}",
    #     sep="    ")

    results = []
    for d in sorted(deprel_to_mask.keys()):
        #if len(deprel_to_mask[d]) < 50: continue
        results.append((d,
            np.mean(deprel_to_mask[d]),
            np.var(deprel_to_mask[d]),
            np.std(deprel_to_mask[d]),
            len(deprel_to_mask[d]),),
        )

    results = sorted(results, key=lambda item: item[1])

    descriptions = {
        "root" : r"Absolute significance score",
        "amod" : r"Adjectival Modifier, e.g. \textit{big} \under{boat}",
        "xcomp" : r"Open Clausal Complement, e.g. I \under{started} to \textit{work}",
        "compound" : r"Compound, e.g. \textit{phone} \under{book}; \textit{ice} \under{cream}",
        "obl" : r"Oblique Nominal, e.g. last \textit{night}, I \under{swam} in the \textit{pool}",
        "obj" : r"Object, e.g. she \under{got} a \textit{gift}",
        "advcl" : r"Adverbial Clause Modifier, e.g. if you \under{know} who did it, you should \textit{say} it",
        "ccomp" : r"Clausal Complement, e.g. he \under{says} that you \textit{like} to swim",
        "nmod" : r"Nominal Modifier, e.g. the \under{office} of the \textit{Chair}",
        "acl" : r"Adnominal Clause), e.g. the \under{issues} as he \textit{sees} them; a simple \under{way} to \textit{get}",
        "mark" : r"Marker, e.g. \textit{before}; \textit{after}; \textit{with}; \textit{without}",
        "conj" : r"Conjunct, e.g. \under{big} and \textit{yellow}",
        "advmod" : r"Adverbial Modifier, e.g. \textit{genetically} \under{modified}, \textit{less} \under{often}",
        "expl" : r"Expletive, e.g. \textit{there} \under{is}, \textit{it} is \under{clear}",
        "fixed" : r"Fixed Multiword Expression, e.g. dogs \under{as} \textit{well} \textit{as} cats; \under{because} \textit{of} you",
        "nsubj" : r"Nominal Subject, e.g. \textit{John} \under{won}",
        "case" : r"Case Marking, e.g. the \under{Chair} \textit{'s} office; the office \textit{of} the \under{Chair}",
        "nummod" : r"Numeric Modifier, e.g. \textit{forty} \under{dollars}, \textit{3} \under{sheep}",
        "aux" : r"Auxiliary, e.g. John \textit{has} \under{died}; he \textit{should} \under{leave}",
        "det" : r"Determiner, e.g. \textit{the} \under{man}",
        "cc" : r"Coordinating Conjunction, e.g. \textit{and} \under{yellow}",
        "cop" : r"Copula, e.g. John \textit{is} the best \under{dancer}; Bill \textit{is} \under{honest}",
        "punct" : r"Punctuation, e.g. \under{Go} home \textit{!}",
        "discourse": r""
    }

    ranks = {}

    iii = 1
    print("deprel, avg, var, std, #")
    for res in results:
        if res[4] <= 100: continue
        if res[0] not in descriptions.keys():
            print("Missing:", res[0])
            continue    

        ranks[res[0]] = iii

        print(res[0],
            f"{res[1]:.2f}",
            #f"{res[2]:.2f}",
            f"{res[3]:.2f}",
            f"{res[4]}",
            f"{descriptions[res[0]]}",
        sep=" & ", end=" \\\\\n")

        iii += 1

    import json

    json_file = open(args.json_path, "w")
    json_file.write(json.dumps(deprel_to_mask) + "\n")
    json_file.write(json.dumps(ranks) + "\n")

    print()

    first_part = deprel_to_mask["det"] + \
                deprel_to_mask["case"] + \
                deprel_to_mask["cop"] + \
                deprel_to_mask["cc"] + \
                deprel_to_mask["punct"] + \
                deprel_to_mask["mark"]

    print(f"det, case, cop, cc, punct, mark & {np.mean(first_part):.2f} & {np.std(first_part):.2f} & {len(first_part)}")

    second_part = deprel_to_mask["advcl"] + \
                deprel_to_mask["acl"] + \
                deprel_to_mask["xcomp"]

    print(f"advcl, acl, xcomp & {np.mean(second_part):.2f} & {np.std(second_part):.2f} & {len(second_part)}")

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
    # plt.savefig("../progress-track/2022-10-11/syntactic_tree.png")