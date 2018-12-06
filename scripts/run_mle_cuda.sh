#!/bin/sh

train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"
dev_src="data/valid.de-en.de"
dev_tgt="data/valid.de-en.en"
test_src="data/test.de-en.de"
test_tgt="data/test.de-en.en"

job_name="iwslt14.ml.512enc.test"
train_log="train."${job_name}".log"
model_name="model."${job_name}
job_file="scripts/train."${job_name}".sh"
decode_file=${job_name}".test.en"

python3 nmt.py \
    --mode train \
    --vocab data/iwslt.vocab.bin \
    --save_to models/${model_name} \
    --valid_niter 2400 \
    --valid_metric ppl \
    --beam_size 5 \
    --batch_size 64 \
    --hidden_size 256 \
    --embed_size 256 \
    --uniform_init 0.1 \
    --clip_grad 5.0 \
    --lr_decay 0.5 \
    --train_src ${train_src} \
    --train_tgt ${train_tgt} \
    --dev_src ${dev_src} \
    --dev_tgt ${dev_tgt} \
    # --dropout 0.2 \

python3 nmt.py \
    --mode test \
    --load_model models/${model_name}.bin \
    --beam_size 5 \
    --decode_max_time_step 100 \
    --save_to_file decode/${decode_file} \
    --test_src ${test_src} \
    --test_tgt ${test_tgt} \

echo "test result" >> logs/${train_log}
perl multi-bleu.perl ${test_tgt} < decode/${decode_file} >> logs/${train_log}
