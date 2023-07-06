import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)

    args, _ = parser.parse_known_args()

    input_stream = sys.stdin if args.input_file == None else open(args.input_file, "r", encoding="utf8")
    output_stream = sys.stdout if args.output_file == None else open(args.output_file, "w", encoding="utf8")

    sentences = input_stream.read().split("\n\n")
    input_stream.close()

    for sentence in sentences:
        items = sentence.split("\n")
        
        ids, forms, lemmas, poss, deprels, heads = [], [], [], [], [], []
        for item in items:
            if item == "": continue
            if item[0] not in "0123456789": continue

            values = item.split("\t")
            ids.append(values[0])
            forms.append(values[1])
            lemmas.append(values[2])
            poss.append(values[3])
            heads.append(values[6])
            deprels.append(values[7])

        ids_c = " ".join(ids); ids = []
        forms_c = " ".join(forms); forms = []
        lemmas_c = " ".join(lemmas); lemmas = []
        poss_c = " ".join(poss); poss = []
        deprels_c = " ".join(deprels); deprels = []
        heads_c = " ".join(heads); heads = []

        print(f"{forms_c}\t{ids_c}\t{lemmas_c}\t{poss_c}\t{heads_c}\t{deprels_c}", file=output_stream)
    
    output_stream.close()
