# Word Importance

This repository contains code snippets used for training models which are described in the paper [Assessing Word Importance Using Models Trained for Semantic Tasks](https://arxiv.org/abs/2305.19689).

## Generating scores

```bash
head -n 10 data/quora-paws-labeled-swap/processed/valid.premise | python ./scripts/masks/evaluation/generate_masks.py --model_path=./interpreters/pi/checkpoint4.ckpt | python ./scripts/masks/evaluation/aggregate_masks.py --scale --float_precision 2
```

Output:
```
What is the difference between coincidence and luck ?   0.04 0.04 0.01 0.18 0.22 1.00 0.09 0.93 0.00
How do I close a pvt ltd company in India ?     0.02 0.05 0.09 0.50 0.03 0.45 1.00 0.67 0.13 0.62 0.00
Who will win if India and China fight now without allies ?      0.29 0.16 0.58 0.44 0.67 0.09 0.94 0.82 0.68 1.00 0.91 0.00
How long could a human survive on just peanut butter and water ?        0.02 0.55 0.39 0.04 0.33 0.73 0.28 0.66 1.00 0.89 0.06 0.77 0.00
What are the importance of mathematical induction ?     0.03 0.03 0.01 0.54 0.08 0.97 1.00 0.00
What is the Sahara , and how do the average temperatures there compare to the ones in the Simpson Desert ?      0.08 0.06 0.04 0.50 0.26 0.09 0.14 0.10 0.05 0.42 0.38 0.23 0.80 0.10 0.05 0.50 0.15 0.05 1.00 0.67 0.00
What does it mean when you dream about someone you haven &apos;t seen for a long time ? 0.03 0.08 0.07 0.14 0.21 0.09 0.83 0.36 0.15 0.11 0.51 1.00 0.30 0.14 0.03 0.78 0.24 0.00
What are some adaptations of the great white shark ?    0.03 0.04 0.09 0.79 0.06 0.02 0.14 0.67 1.00 0.00
How much app partition size does Lenovo K3 Note have ?  0.03 0.31 0.43 0.31 0.46 0.11 1.00 0.78 0.41 0.12 0.00
What is the best smartphone app ?       0.12 0.10 0.06 0.20 1.00 0.98 0.00
```

## Datasets

Datasets can be downloaded from the following [link](http://ufallab.ms.mff.cuni.cz/~javorsky/data/). Store the `data` folder here to make everything work.

## Training

Underlying models (NLI and PI)

```bash
./train.sh
```

Masks

```bash
python scripts/masks/train/run_nli_diffmask.py
```

## Semantic models

Trained models can be downloaded from the following [link](http://ufallab.ms.mff.cuni.cz/~javorsky/models/). Store the `models` folder here to make everything work.

## Interpreters

Trained interpreters can be downloaded from the following [link](http://ufallab.ms.mff.cuni.cz/~javorsky/interpreters/). Store the `interpreters` folder here to make everything work.

## Installation

### Automatized installation

```bash
./setup.sh <python_path> <virtual_env_name>
```

### Manual installation

#### Virtual environment

```bash
/opt/python/3.8.14/bin/python3 -m venv env3.8.14
```

```bash
source env3.8.14/bin/activate
```

```bash
pip3 install --upgrade pip setuptools wheel
```

#### Diffmask

```bash
git clone https://github.com/nicola-decao/diffmask.git
```

```
cd diffmask

pip install -r requirements.txt

python setup.py install

cd ..
```

```bash
mkdir env3.8.14/lib/python3.8/site-packages/diffmask
cp -r diffmask/diffmask env3.8.14/lib/python3.8/site-packages/diffmask
rm -r diffmask
```

```bash
ln -vfns $(pwd)/scripts/masks/train/nli_classification_diffmask.py env3.8.14/lib/python3.8/site-packages/diffmask/models
ln -vfns $(pwd)/scripts/masks/train/nli_classification.py env3.8.14/lib/python3.8/site-packages/diffmask/models
ln -vfns $(pwd)/scripts/masks/train/gates.py env3.8.14/lib/python3.8/site-packages/diffmask/models
ln -vfns $(pwd)/scripts/masks/train/sentiment_classification_sst_diffmask.py env3.8.14/lib/python3.8/site-packages/diffmask/models
ln -vfns $(pwd)/scripts/masks/train/sentiment_classification_sst.py env3.8.14/lib/python3.8/site-packages/diffmask/models
ln -vfns $(pwd)/scripts/masks/train/callbacks.py env3.8.14/lib/python3.8/site-packages/diffmask/utils
ln -vfns $(pwd)/scripts/masks/train/plot.py env3.8.14/lib/python3.8/site-packages/diffmask/utils

```

#### Fairseq

```bash
pip install fairseq
```

```bash
ln -vfns $(pwd)/scripts/nli/train/nli_generator.py env3.8.14/lib/python3.8/site-packages/fairseq
ln -vfns $(pwd)/scripts/nli/train/nli_classification.py env3.8.14/lib/python3.8/site-packages/fairseq/tasks
ln -vfns $(pwd)/scripts/nli/train/nli_classifier.py env3.8.14/lib/python3.8/site-packages/fairseq/models
ln -vfns $(pwd)/scripts/nli/train/nli_loss.py env3.8.14/lib/python3.8/site-packages/fairseq/criterions
ln -vfns $(pwd)/scripts/nli/train/nli_loss.py env3.8.14/lib/python3.8/site-packages/fairseq/criterions
```

#### Torch

We need a specific version of torch.

```bash
pip install torch==1.6.0
```

# Tools

## Interactive Gradio App

Install gradio (`pip install gradio`) and run `python gradio_example.py` and test the models using an interactive web-based application.

## NLI

### Preprocessing

Reads from the standard input and capitalizes first letter, adds full-stop if missing, and tokenizes.

Example:

```bash
cat data/msq-nli/raw/train.premise | ./scripts/nli/preprocess/tokenize.sh > data/msq-nli/tokenized/train.premise
cat data/msq-nli/raw/valid.premise | ./scripts/nli/preprocess/tokenize.sh > data/msq-nli/tokenized/valid.premise
cat data/msq-nli/raw/test.premise | ./scripts/nli/preprocess/tokenize.sh > data/msq-nli/tokenized/test.premise
cat data/msq-nli/raw/train.hypothesis | ./scripts/nli/preprocess/tokenize.sh > data/msq-nli/tokenized/train.hypothesis
cat data/msq-nli/raw/valid.hypothesis | ./scripts/nli/preprocess/tokenize.sh > data/msq-nli/tokenized/valid.hypothesis
cat data/msq-nli/raw/test.hypothesis | ./scripts/nli/preprocess/tokenize.sh > data/msq-nli/tokenized/test.hypothesis
cp data/msq-nli/raw/train.label data/msq-nli/tokenized/train.label
cp data/msq-nli/raw/valid.label data/msq-nli/tokenized/valid.label
cp data/msq-nli/raw/test.label data/msq-nli/tokenized/test.label
```

## Masks

### `generate_masks.py`

Given a model `model.ckpt` the script generates attributes for each input sentence. The output is the tab-separated original line and list of attributes for each input token. The format of input should be the same as for training the model `model.ckpt`.

In more detail, the output is:
- `line <tab> list_of_attributes` for each input line
- `list_of_attributes` = `attributes_for_token_1 <tab> attributes_for_token_2 <tab> ... <tab> attributes_for_token_N` for each token 1-N of the input line
- `attributes_for_token_x` = `attribute_1 <space> attribute_2 <space> ... <space> attribute_H` for each hidden layer 1-H of the encoder

```bash
python generate_masks.py --model_path model.ckpt < input_file.txt > output_file.txt
```
```bash
python generate_masks.py --model_path model.ckpt --input_file input_file.txt --output_file output_file.txt
```

Example:

```bash
echo "The sisters are hugging goodbye while holding to go packages after just eating lunch ." | python generate_masks.py --float_precision 2 2>/dev/null
```

```bash
The sisters are hugging goodbye while holding to go packages after just eating lunch .  0.97 0.56 0.34 0.32 0.30 0.28 0.26 0.25 1.00 0.64 0.64 0.53 0.49 0.46 0.43 0.42 0.81 0.47 0.30 0.24 0.22 0.20 0.19 0.18       0.94 0.54 0.41 0.33 0.30 0.28 0.26 0.25 0.99 0.57 0.57 0.47 0.44 0.41 0.38 0.37 0.92 0.53 0.42 0.35 0.32 0.30 0.28 0.26 0.90 0.52 0.40 0.35 0.33 0.30 0.28 0.27 0.94 0.54 0.53 0.45 0.42 0.38 0.36 0.34       0.98 0.57 0.39 0.35 0.33 0.31 0.29 0.27 0.99 0.57 0.47 0.44 0.42 0.39 0.37 0.35 1.00 0.58 0.46 0.41 0.39 0.37 0.35 0.33 1.00 0.58 0.51 0.49 0.47 0.44 0.42 0.40 0.99 0.57 0.44 0.42 0.41 0.38 0.36 0.34       1.00 0.59 0.43 0.39 0.37 0.34 0.32 0.31 0.65 0.38 0.22 0.19 0.18 0.16 0.15 0.14
```

### `aggregate_masks.py`

Given the output from `generate_masks.py` the scripts aggregates attributes for each token. `--mode` specifies the type of computation: `average` or `last` value.

The script can be called either with input/output redirection or from files. The output is the original sentence and the list of scores for each token. The sentence and the list are separated by `<tab>`.

Example:

```bash
echo "The sisters are hugging goodbye while holding to go packages after just eating lunch ." | python generate_masks.py 2>/dev/null | python ../scripts/masks/evaluation/aggregate_masks.py --float_precision 2
```

```bash
The sisters are hugging goodbye while holding to go packages after just eating lunch .  0.40 1.00 0.14 0.38 0.82 0.45 0.47 0.74 0.49 0.77 0.70 0.94 0.73 0.62 0.00
```

### `filter_sentences.py`

Given the output from `aggregate_masks.py` the script filters out tokens according to the `--threshold` between 0-1 and `--mode` either `length`, `mask` or `random`.
- `length`: Tokens are sorted according to their score and those with the lowest values are discarded based on `--threshold`.
- `mask`: Tokens with a lower score than `--threshold` are discarded.
- `random`: Tokens are discarded randomly. The number of tokens is the same as if the `length` node was used.

The output is a triple of `sentence`, `filtered_sentence`, `displayed_filter`.

Example:

```bash
echo "The sisters are hugging goodbye while holding to go packages after just eating lunch ." | python generate_masks.py --float_precision 5 2>/dev/null | python ../scripts/masks/evaluation/aggregate_masks.py | python filter_sentences.py
```

```bash
The sisters are hugging goodbye while holding to go packages after just eating lunch .  The sisters goodbye while holding to go packages after just eating lunch        The sisters [are] [hugging] goodbye while holding to go packages after just eating lunch [.]
```

### `generate_conllu.sh`

Given the input file as a command line argument, the script generates lemmas, pos and trees for each input sentence. The output format is `conllu`.

The script reads from the file which is given as a command line argument. Otherwise, it expects one line on the standard input.

Examples:

```bash
./scripts/masks/evaluation/generate_conllu.sh input_file.txt
```

```bash
echo "The sisters are hugging goodbye while holding to go packages after just eating lunch ." | ./scripts/masks/evaluation/generate_conllu.sh
```

```bash
# generator = UDPipe 2, https://lindat.mff.cuni.cz/services/udpipe
# udpipe_model = english-ewt-ud-2.10-220711
# udpipe_model_licence = CC BY-NC-SA
# newdoc
# newpar
# sent_id = 1
# text = The sisters are hugging goodbye while holding to go packages after just eating lunch .
1       The     the     DET     DT      Definite=Def|PronType=Art       2       det     _       _
2       sisters sister  NOUN    NNS     Number=Plur     4       nsubj   _       _
3       are     be      AUX     VBP     Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin   4       aux     _       _
4       hugging hug     VERB    VBG     Tense=Pres|VerbForm=Part        0       root    _       _
5       goodbye goodbye NOUN    NN      Number=Sing     4       obj     _       _
6       while   while   SCONJ   IN      _       7       mark    _       _
7       holding hold    VERB    VBG     VerbForm=Ger    4       advcl   _       _
8       to      to      PART    TO      _       9       mark    _       _
9       go      go      VERB    VB      VerbForm=Inf    10      compound        _       _
10      packages        package NOUN    NNS     Number=Plur     7       obj     _       _
11      after   after   SCONJ   IN      _       13      mark    _       _
12      just    just    ADV     RB      _       13      advmod  _       _
13      eating  eat     VERB    VBG     VerbForm=Ger    7       advcl   _       _
14      lunch   lunch   NOUN    NN      Number=Sing     13      obj     _       _
15      .       .       PUNCT   .       _       4       punct   _       SpaceAfter=No
```

### `process_conllu.py`

Given the output from the `generate_conllu.sh`, the script processes to the format of a `<tab>`-separated list of items in the following order:
- `original tokens`: The original sentence. 
- `ids`: The list of token ids, counted from 1.
- `lemmas`: The lowercased list of lemmas.
- `POS`: The list of part of speech tags.
- `heads`: The list of ids of parent nodes in the syntactic tree.
- `deprels`: The list of dependency relations between parent nodes and their children in the syntactic tree.

Example:

```bash
echo "The sisters are hugging goodbye while holding to go packages after just eating lunch ." | ./scripts/masks/evaluation/generate_conllu.sh | python scripts/masks/evaluation/process_conllu.py
```

```bash
The sisters are hugging goodbye while holding to go packages after just eating lunch .  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15     the sister be hug goodbye while hold to go package after just eat lunch . DET NOUN AUX VERB NOUN SCONJ VERB PART VERB NOUN SCONJ ADV VERB NOUN PUNCT      2 4 4 0 4 7 4 9 10 7 13 13 7 13 4 det nsubj aux root obj mark advcl mark compound obj mark advmod advcl obj punct
```

From the [udpipe web service](https://lindat.mff.cuni.cz/services/udpipe/):

## CREDITS

```
@inproceedings{javorsky-etal-2023-assessing,
    title = "Assessing Word Importance Using Models Trained for Semantic Tasks",
    author = "Javorsk{\'y}, D{\'a}vid  and
      Bojar, Ond{\v{r}}ej  and
      Yvon, Fran{\c{c}}ois",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.563",
    doi = "10.18653/v1/2023.findings-acl.563",
    pages = "8846--8856",
    abstract = "Many NLP tasks require to automatically identify the most significant words in a text. In this work, we derive word significance from models trained to solve semantic task: Natural Language Inference and Paraphrase Identification. Using an attribution method aimed to explain the predictions of these models, we derive importance scores for each input token. We evaluate their relevance using a so-called cross-task evaluation: Analyzing the performance of one model on an input masked according to the other model{'}s weight, we show that our method is robust with respect to the choice of the initial task. Additionally, we investigate the scores from the syntax point of view and observe interesting patterns, e.g. words closer to the root of a syntactic tree receive higher importance scores. Altogether, these observations suggest that our method can be used to identify important words in sentences without any explicit word importance labeling in training.",
}
```