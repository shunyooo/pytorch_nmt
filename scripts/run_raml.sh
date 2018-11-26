#!/bin/sh

train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"
dev_src="data/valid.de-en.de"
dev_tgt="data/valid.de-en.en"
test_src="data/test.de-en.de"
test_tgt="data/test.de-en.en"

now=`date "+%Y%m%dT%H%M%S"`
job_name="iwslt14.raml.512enc.corrupt_ngram.t0.600.test"
train_log="train."${job_name}${now}".log"
model_name="model."${job_name}${now}
job_file="scripts/train."${job_name}${now}".sh"
decode_file=${job_name}${now}".test.en"

log_file="./logs/"${now}"_stdout.log"
train_log_file="./logs/"${now}"_train.log"
validation_log_file="./logs/"${now}"_validation.log"
temp="0.6"

echo save model to models/${model_name}
echo log to ${log_file}

python3 -u\
     -m ipdb nmt.py \
    --mode raml_train \
    --vocab data/iwslt.vocab.bin \
    --save_to models/${model_name} \
    --valid_niter 15400 \
    --valid_metric bleu \
    --beam_size 5 \
    --batch_size 10 \
    --sample_size 10 \
    --hidden_size 256 \
    --embed_size 256 \
    --uniform_init 0.1 \
    --clip_grad 5.0 \
    --lr_decay 0.5 \
    --temp ${temp} \
    --train_src ${train_src} \
    --train_tgt ${train_tgt} \
    --dev_src ${dev_src} \
    --dev_tgt ${dev_tgt} \
    --raml_sample_file ./tmp/samples.txt \
    --log_every 50 \
    --train_log_file ${train_log_file} \
    --validation_log_file ${validation_log_file} \
    #    -m ipdb nmt.py \
    # --dropout 0.2 \
    # --raml_sample_file data/samples.corrupt_ngram.bleu_score.txt \

#python3 nmt.py \
#    --mode test \
#    --load_model models/${model_name}.bin \
#    --beam_size 5 \
#    --decode_max_time_step 100 \
#    --save_to_file decode/${decode_file} \
#    --test_src ${test_src} \
#    --test_tgt ${test_tgt}
#
#echo "test result" >> logs/${train_log}
#perl multi-bleu.perl ${test_tgt} < decode/${decode_file} >> logs/${train_log}
