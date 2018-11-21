#!/bin/sh

train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"

python3 process_samples.py \
    --mode sample_ngram \
    --vocab data/vocab2.bin \
    --src ${train_src} \
    --tgt ${train_tgt} \
    --output ./tmp/samples.txt \
    --smooth_bleu