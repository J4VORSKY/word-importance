# Word Importance

This repository contains code snippets used for training models which are described in the paper [Assessing Word Importance Using Models Trained for Semantic Tasks](https://arxiv.org/abs/2305.19689).

## Installation

Python 3.8.

```bash
./setup.sh <python_path> <virtual_env_name>
```

## Datasets

Datasets can be downloaded from the following [link](http://ufallab.ms.mff.cuni.cz/~javorsky/data/).

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

Trained models can be downloaded from the following [link](http://ufallab.ms.mff.cuni.cz/~javorsky/models/).

## Interpreters

Trained interpreters can be downloaded from the following [link](http://ufallab.ms.mff.cuni.cz/~javorsky/interpreters/).

## CREDITS

```
TBA
```