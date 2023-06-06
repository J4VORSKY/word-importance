#!/bin/bash

PYT=$1
ENV=$2

$PYT -m venv $ENV

source $ENV/bin/activate

pip3 install --upgrade pip setuptools wheel

# Fairseq

pip install fairseq

pip install torch==1.6.0

ln -vfns /lnet/troja/work/people/javorsky/text-compression/scripts/nli/train/nli_generator.py $ENV/lib/python3.8/site-packages/fairseq
ln -vfns /lnet/troja/work/people/javorsky/text-compression/scripts/nli/train/nli_classification.py $ENV/lib/python3.8/site-packages/fairseq/tasks
ln -vfns /lnet/troja/work/people/javorsky/text-compression/scripts/nli/train/nli_classifier.py $ENV/lib/python3.8/site-packages/fairseq/models
ln -vfns /lnet/troja/work/people/javorsky/text-compression/scripts/nli/train/nli_loss.py $ENV/lib/python3.8/site-packages/fairseq/criterions

# Diffmask

git clone https://github.com/nicola-decao/diffmask.git

cd diffmask
python setup.py install
cd ..

mkdir $ENV/lib/python3.8/site-packages/diffmask
cp -r diffmask/diffmask $ENV/lib/python3.8/site-packages/diffmask
rm diffmask

ln -vfns /lnet/troja/work/people/javorsky/text-compression/scripts/masks/train/nli_classification_diffmask.py $ENV/lib/python3.8/site-packages/diffmask/models
ln -vfns /lnet/troja/work/people/javorsky/text-compression/scripts/masks/train/nli_classification.py $ENV/lib/python3.8/site-packages/diffmask/models
ln -vfns /lnet/troja/work/people/javorsky/text-compression/scripts/masks/train/gates.py $ENV/lib/python3.8/site-packages/diffmask/models
ln -vfns /lnet/troja/work/people/javorsky/text-compression/scripts/masks/train/sentiment_classification_sst_diffmask.py $ENV/lib/python3.8/site-packages/diffmask/models
ln -vfns /lnet/troja/work/people/javorsky/text-compression/scripts/masks/train/sentiment_classification_sst.py $ENV/lib/python3.8/site-packages/diffmask/models
ln -vfns /lnet/troja/work/people/javorsky/text-compression/scripts/masks/train/callbacks.py $ENV/lib/python3.8/site-packages/diffmask/utils
ln -vfns /lnet/troja/work/people/javorsky/text-compression/scripts/masks/train/plot.py $ENV/lib/python3.8/site-packages/diffmask/utils
