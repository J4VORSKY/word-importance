#!/bin/bash 

LAYERS=6
HEADS=8
FFN=1024
EMB=512

SAVE_PATH=models

MODEL=$LAYERS-l.$HEADS-h.$FFN-ffn.$EMB-emb.nli
DATA_PATH=data/msq-nli

mkdir -p $SAVE_PATH/$MODEL

source ./env3.8.14/bin/activate

# --model_type: [nli|paraphrase]

CUDA_LAUNCH_BLOCKING=1 fairseq-train $DATA_PATH \
  --task nli_classification \
  --arch nli_bi_encoder \
  --criterion nli_loss \
  --model_type nli \
  --encoder-attention-heads $HEADS \
  --encoder-ffn-embed-dim $FFN \
  --encoder-embed-dim $EMB \
  --encoder-layers $LAYERS \
  --optimizer adam \
  --max-tokens 65536  \
  --validate-interval-updates 200 \
  --adam-betas '(0.90, 0.98)' \
  --clip-norm 0.0 \
  --lr 0.0003 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 500 \
  --dropout 0.1 \
  --weight-decay 0.00001 \
  --save-dir $SAVE_PATH/$MODEL \
  --tensorboard-logdir $SAVE_PATH/$MODEL \
  --log-file $SAVE_PATH/$MODEL/logs.txt \
  --log-interval 200 \
  --keep-best-checkpoints 1 \
  --best-checkpoint-metric acc \
  --maximize-best-checkpoint-metric \
  --max-epoch 6 \
  --num-workers 5 \
  --skip-invalid-size-inputs-valid-test \
  --fp16
