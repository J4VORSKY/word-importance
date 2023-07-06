#!/bin/bash -v

PYT=$1
ENV=$2

CUR_DIR=$(pwd)

$PYT -m venv $ENV

source $ENV/bin/activate

pip3 install --upgrade pip setuptools wheel

# Fairseq

pip install fairseq

ln -vfns $CUR_DIR/scripts/nli/train/nli_generator.py $ENV/lib/python3.8/site-packages/fairseq
ln -vfns $CUR_DIR/scripts/nli/train/nli_classification.py $ENV/lib/python3.8/site-packages/fairseq/tasks
ln -vfns $CUR_DIR/scripts/nli/train/nli_classifier.py $ENV/lib/python3.8/site-packages/fairseq/models
ln -vfns $CUR_DIR/scripts/nli/train/nli_loss.py $ENV/lib/python3.8/site-packages/fairseq/criterions

# Torch

pip install torch==1.6.0

# Diffmask

git clone https://github.com/nicola-decao/diffmask.git

cd diffmask
python setup.py install
cd ..

mkdir $ENV/lib/python3.8/site-packages/diffmask
cp -r diffmask/diffmask $ENV/lib/python3.8/site-packages
rm -rf diffmask

ln -vfns $CUR_DIR/scripts/masks/train/nli_classification_diffmask.py $ENV/lib/python3.8/site-packages/diffmask/models
ln -vfns $CUR_DIR/scripts/masks/train/nli_classification.py $ENV/lib/python3.8/site-packages/diffmask/models
ln -vfns $CUR_DIR/scripts/masks/train/gates.py $ENV/lib/python3.8/site-packages/diffmask/models
ln -vfns $CUR_DIR/scripts/masks/train/sentiment_classification_sst_diffmask.py $ENV/lib/python3.8/site-packages/diffmask/models
ln -vfns $CUR_DIR/scripts/masks/train/sentiment_classification_sst.py $ENV/lib/python3.8/site-packages/diffmask/models
ln -vfns $CUR_DIR/scripts/masks/train/callbacks.py $ENV/lib/python3.8/site-packages/diffmask/utils
ln -vfns $CUR_DIR/scripts/masks/train/plot.py $ENV/lib/python3.8/site-packages/diffmask/utils
