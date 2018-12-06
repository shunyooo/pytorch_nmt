# -*- coding: utf-8 -*-
from __future__ import print_function

import re

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
from torch import optim
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import time
import numpy as np
from collections import defaultdict, Counter, namedtuple
from itertools import chain, islice
import argparse, os, sys

from nmt_model import NMT
from nmt_util import to_input_variable
from util import read_corpus, data_iter, batch_slice
from vocab import Vocab, VocabEntry
from process_samples import generate_hamming_distance_payoff_distribution
import math

import slack


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='use gpu')
    parser.add_argument('--mode', choices=['train', 'raml_train', 'test', 'sample', 'prob', 'interactive'],
                        default='train', help='run mode')
    parser.add_argument('--vocab', type=str, help='path of the serialized vocabulary')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--sample_size', default=10, type=int, help='sample size')
    parser.add_argument('--embed_size', default=256, type=int, help='size of word embeddings')
    parser.add_argument('--hidden_size', default=256, type=int, help='size of LSTM hidden states')
    parser.add_argument('--dropout', default=0., type=float, help='dropout rate')

    parser.add_argument('--train_src', type=str, help='path to the training source file')
    parser.add_argument('--train_tgt', type=str, help='path to the training target file')
    parser.add_argument('--dev_src', type=str, help='path to the dev source file')
    parser.add_argument('--dev_tgt', type=str, help='path to the dev target file')
    parser.add_argument('--test_src', type=str, help='path to the test source file')
    parser.add_argument('--test_tgt', type=str, help='path to the test target file')

    parser.add_argument('--decode_max_time_step', default=200, type=int, help='maximum number of time steps used '
                                                                              'in decoding and sampling')

    parser.add_argument('--valid_niter', default=500, type=int, help='every n iterations to perform validation')
    parser.add_argument('--valid_metric', default='bleu', choices=['bleu', 'ppl', 'word_acc', 'sent_acc'],
                        help='metric used for validation')
    parser.add_argument('--log_every', default=50, type=int, help='every n iterations to log training statistics')
    parser.add_argument('--train_log_file', default='model', type=str, help='学習時のスコア保存')
    parser.add_argument('--validation_log_file', default='model', type=str, help='テスト時のスコア保存')
    parser.add_argument('--load_model', default=None, type=str, help='load a pre-trained model')
    parser.add_argument('--save_to', default='model', type=str, help='save trained model to')
    parser.add_argument('--save_model_after', default=2, help='save the model only after n validation iterations')
    parser.add_argument('--save_to_file', default=None, type=str, help='if provided, save decoding results to file')
    parser.add_argument('--save_nbest', default=False, action='store_true', help='save nbest decoding results')
    parser.add_argument('--patience', default=5, type=int, help='training patience')
    parser.add_argument('--uniform_init', default=None, type=float,
                        help='if specified, use uniform initialization for all parameters')
    parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    parser.add_argument('--max_niter', default=-1, type=int, help='maximum number of training iterations')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.5, type=float,
                        help='decay learning rate if the validation performance drops')

    # raml training
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--temp', default=0.85, type=float, help='temperature in reward distribution')
    parser.add_argument('--raml_sample_mode', default='pre_sample',
                        choices=['pre_sample', 'hamming_distance', 'hamming_distance_impt_sample'],
                        help='sample mode when using RAML')
    parser.add_argument('--raml_sample_file', type=str, help='path to the sampled targets')
    parser.add_argument('--raml_bias_groundtruth', action='store_true', default=False,
                        help='make sure ground truth y* is in samples')

    parser.add_argument('--smooth_bleu', action='store_true', default=False,
                        help='smooth sentence level BLEU score.')

    parser.add_argument('--notify_slack', action='store_true', default=True,
                        help='notify slack')

    # TODO: greedy sampling is still buggy!
    parser.add_argument('--sample_method', default='random', choices=['random', 'greedy'])

    args = parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args


def evaluate_loss(model, data, crit):
    model.eval()
    cum_loss = 0.
    cum_tgt_words = 0.
    for src_sents, tgt_sents in data_iter(data, batch_size=args.batch_size, shuffle=False):
        pred_tgt_word_num = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
        src_sents_len = [len(s) for s in src_sents]

        src_sents_var = to_input_variable(src_sents, model.vocab.src, cuda=args.cuda, is_test=True)
        tgt_sents_var = to_input_variable(tgt_sents, model.vocab.tgt, cuda=args.cuda, is_test=True)

        # (tgt_sent_len, batch_size, tgt_vocab_size)
        scores = model(src_sents_var, src_sents_len, tgt_sents_var[:-1])
        loss = crit(scores.view(-1, scores.size(2)), tgt_sents_var[1:].view(-1))

        cum_loss += loss.item()
        cum_tgt_words += pred_tgt_word_num

    loss = cum_loss / cum_tgt_words
    return loss


def init_training(args):
    from functools import partial
    import pickle
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    # model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
    vocab = torch.load(args.vocab, map_location=lambda storage, loc: storage, pickle_module=pickle)

    model = NMT(args, vocab)
    model.train()

    if args.uniform_init:
        print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-args.uniform_init, args.uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0
    nll_loss = nn.NLLLoss(weight=vocab_mask, reduction='sum')
    cross_entropy_loss = nn.CrossEntropyLoss(weight=vocab_mask, reduction='sum')

    if args.cuda:
        model = nn.DataParallel(model).cuda()
        nll_loss = nll_loss.cuda()
        cross_entropy_loss = cross_entropy_loss.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    return vocab, model, optimizer, nll_loss, cross_entropy_loss


def train(args):
    train_data_src = read_corpus(args.train_src, source='src')
    train_data_tgt = read_corpus(args.train_tgt, source='tgt')

    dev_data_src = read_corpus(args.dev_src, source='src')
    dev_data_tgt = read_corpus(args.dev_tgt, source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    vocab, model, optimizer, nll_loss, cross_entropy_loss = init_training(args)

    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = cum_batches = report_examples = epoch = valid_num = best_model_iter = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    with open(args.train_log_file, "w") as train_output, open(args.validation_log_file, "w") as validation_output:

        while True:
            epoch += 1
            for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
                train_iter += 1

                src_sents_var = to_input_variable(src_sents, vocab.src, cuda=args.cuda)
                tgt_sents_var = to_input_variable(tgt_sents, vocab.tgt, cuda=args.cuda)

                batch_size = len(src_sents)
                src_sents_len = [len(s) for s in src_sents]
                pred_tgt_word_num = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`

                optimizer.zero_grad()

                # (tgt_sent_len, batch_size, tgt_vocab_size)
                scores = model(src_sents_var, src_sents_len, tgt_sents_var[:-1])

                word_loss = cross_entropy_loss(scores.view(-1, scores.size(2)), tgt_sents_var[1:].view(-1))
                loss = word_loss / batch_size
                word_loss_val = word_loss.item()
                loss_val = loss.item()

                loss.backward()
                # clip gradient
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

                report_loss += word_loss_val
                cum_loss += word_loss_val
                report_tgt_words += pred_tgt_word_num
                cum_tgt_words += pred_tgt_word_num
                report_examples += batch_size
                cum_examples += batch_size
                cum_batches += batch_size

                if train_iter % args.log_every == 0:
                    _log = 'epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                           'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                              report_loss / report_examples,
                                                                                              np.exp(
                                                                                                  report_loss / report_tgt_words),
                                                                                              cum_examples,
                                                                                              report_tgt_words / (
                                                                                                      time.time() - train_time),
                                                                                              time.time() - begin_time)
                    print(_log)
                    print(_log, file=train_output)

                    train_time = time.time()
                    report_loss = report_tgt_words = report_examples = 0.

                # perform validation
                if train_iter % args.valid_niter == 0:
                    print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                                 cum_loss / cum_batches,
                                                                                                 np.exp(
                                                                                                     cum_loss / cum_tgt_words),
                                                                                                 cum_examples),
                          file=sys.stderr)

                    cum_loss = cum_batches = cum_tgt_words = 0.
                    valid_num += 1

                    print('begin validation ...', file=sys.stderr)
                    model.eval()

                    # compute dev. ppl and bleu

                    dev_loss = evaluate_loss(model, dev_data, cross_entropy_loss)
                    dev_ppl = np.exp(dev_loss)

                    if args.valid_metric in ['bleu', 'word_acc', 'sent_acc']:
                        dev_hyps = decode(model, dev_data)
                        dev_hyps = [hyps[0] for hyps in dev_hyps]
                        if args.valid_metric == 'bleu':
                            valid_metric = get_bleu([tgt for src, tgt in dev_data], dev_hyps)
                        else:
                            valid_metric = get_acc([tgt for src, tgt in dev_data], dev_hyps, acc_type=args.valid_metric)
                        _log = 'validation: iter %d, dev. ppl %f, dev. %s %f' % (
                            train_iter, dev_ppl, args.valid_metric, valid_metric)
                        print(_log, file=sys.stderr)
                        print(_log, file=validation_output)
                        if args.notify_slack:
                            slack.post(_log)

                    else:
                        valid_metric = -dev_ppl
                        print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl),
                              file=sys.stderr)

                    model.train()

                    is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                    is_better_than_last = len(hist_valid_scores) == 0 or valid_metric > hist_valid_scores[-1]
                    hist_valid_scores.append(valid_metric)

                    if valid_num > args.save_model_after:
                        model_file = args.save_to + '.iter%d.bin' % train_iter
                        print('save model to [%s]' % model_file, file=sys.stderr)
                        model.save(model_file)

                    if (not is_better_than_last) and args.lr_decay:
                        lr = optimizer.param_groups[0]['lr'] * args.lr_decay
                        print('decay learning rate to %f' % lr, file=sys.stderr)
                        optimizer.param_groups[0]['lr'] = lr

                    if is_better:
                        patience = 0
                        best_model_iter = train_iter

                        if valid_num > args.save_model_after:
                            print('save currently the best model ..', file=sys.stderr)
                            model_file_abs_path = os.path.abspath(model_file)
                            symlin_file_abs_path = os.path.abspath(args.save_to + '.bin')
                            os.system('ln -sf %s %s' % (model_file_abs_path, symlin_file_abs_path))
                    else:
                        patience += 1
                        print('hit patience %d' % patience, file=sys.stderr)
                        if patience == args.patience:
                            print('early stop!', file=sys.stderr)
                            print('the best model is from iteration [%d]' % best_model_iter, file=sys.stderr)
                            exit(0)


def read_raml_train_data(data_file, temp):
    train_data = dict()
    num_pattern = re.compile('^(\d+) samples$')
    with open(data_file) as f:
        while True:
            line = f.readline()
            if line is None or line == '':
                break

            assert line.startswith('***')

            src_sent = f.readline()[len('source: '):].strip()
            tgt_num = int(num_pattern.match(f.readline().strip()).group(1))
            tgt_samples = []
            tgt_scores = []
            for i in range(tgt_num):
                d = f.readline().strip().split(' ||| ')
                if len(d) < 2:
                    continue

                tgt_sent = d[0].strip()
                bleu_score = float(d[1])
                tgt_samples.append(tgt_sent)
                tgt_scores.append(bleu_score / temp)

            tgt_scores = np.exp(tgt_scores)
            tgt_scores = tgt_scores / np.sum(tgt_scores)

            tgt_entry = list(zip(tgt_samples, tgt_scores))
            train_data[src_sent] = tgt_entry

            line = f.readline()

    return train_data

def train_raml(args):
    tau = args.temp

    train_data_src = read_corpus(args.train_src, source='src')
    train_data_tgt = read_corpus(args.train_tgt, source='tgt')
    train_data = list(zip(train_data_src, train_data_tgt))

    dev_data_src = read_corpus(args.dev_src, source='src')
    dev_data_tgt = read_corpus(args.dev_tgt, source='tgt')
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    vocab, model, optimizer, nll_loss, cross_entropy_loss = init_training(args)

    if args.raml_sample_mode == 'pre_sample':
        # dict of (src, [tgt: (sent, prob)])
        print('read in raml training data...', file=sys.stderr, end='')
        begin_time = time.time()
        raml_samples = read_raml_train_data(args.raml_sample_file, temp=tau)
        print('done[%d s].' % (time.time() - begin_time))
    elif args.raml_sample_mode.startswith('hamming_distance'):
        print('sample from hamming distance payoff distribution')
        payoff_prob, Z_qs = generate_hamming_distance_payoff_distribution(max(len(sent) for sent in train_data_tgt),
                                                                          vocab_size=len(vocab.tgt) - 3,
                                                                          tau=tau)

    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    report_weighted_loss = cum_weighted_loss = 0
    cum_examples = cum_batches = report_examples = epoch = valid_num = best_model_iter = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin RAML training')

    if args.notify_slack:
        slack.post(f"""
        {args}
        begin RAML training
        """)


    # smoothing function for BLEU
    sm_func = None
    if args.smooth_bleu:
        sm_func = SmoothingFunction().method3

    with open(args.train_log_file, "w") as train_output, open(args.validation_log_file, "w") as validation_output:

        while True:
            epoch += 1
            for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
                train_iter += 1

                raml_src_sents = []
                raml_tgt_sents = []
                raml_tgt_weights = []

                if args.raml_sample_mode == 'pre_sample':
                    for src_sent in src_sents:
                        sent = ' '.join(src_sent)
                        tgt_samples_all = raml_samples[sent]
                        # print(f'src_sent: "{sent}", target_samples_all: {len(list(tgt_samples_all))}')
                        if args.sample_size >= len(list(tgt_samples_all)):
                            tgt_samples = tgt_samples_all
                        else:
                            tgt_samples_id = np.random.choice(range(1, len(list(tgt_samples_all))),
                                                              size=args.sample_size - 1, replace=False)
                            tgt_samples = [tgt_samples_all[0]] + [tgt_samples_all[i] for i in
                                                                  tgt_samples_id]  # make sure the ground truth y* is in the samples

                        raml_src_sents.extend([src_sent] * len(list(tgt_samples)))
                        raml_tgt_sents.extend([['<s>'] + sent.split(' ') + ['</s>'] for sent, weight in tgt_samples])
                        raml_tgt_weights.extend([weight for sent, weight in tgt_samples])
                elif args.raml_sample_mode in ['hamming_distance', 'hamming_distance_impt_sample']:
                    for src_sent, tgt_sent in zip(src_sents, tgt_sents):
                        tgt_samples =  []  # make sure the ground truth y* is in the samples
                        tgt_sent_len = len(tgt_sent) - 3  # remove <s> and </s> and ending period .
                        tgt_ref_tokens = tgt_sent[1:-1]
                        bleu_scores = []
                        # print('y*: %s' % ' '.join(tgt_sent))
                        # sample an edit distances
                        e_samples = np.random.choice(range(tgt_sent_len + 1), p=payoff_prob[tgt_sent_len],
                                                     size=args.sample_size, replace=True)

                        # make sure the ground truth y* is in the samples
                        if args.raml_bias_groundtruth and (not 0 in e_samples):
                            e_samples[0] = 0

                        for i, e in enumerate(e_samples):
                            if e > 0:
                                # sample a new tgt_sent $y$
                                old_word_pos = np.random.choice(range(1, tgt_sent_len + 1), size=e, replace=False)
                                new_words = [vocab.tgt.id2word[wid] for wid in
                                             np.random.randint(3, len(vocab.tgt), size=e)]
                                new_tgt_sent = list(tgt_sent)
                                for pos, word in zip(old_word_pos, new_words):
                                    new_tgt_sent[pos] = word
                            else:
                                new_tgt_sent = list(tgt_sent)

                            # if enable importance sampling, compute bleu score
                            if args.raml_sample_mode == 'hamming_distance_impt_sample':
                                if e > 0:
                                    # remove <s> and </s>
                                    bleu_score = sentence_bleu([tgt_ref_tokens], new_tgt_sent[1:-1],
                                                               smoothing_function=sm_func)
                                    bleu_scores.append(bleu_score)
                                else:
                                    bleu_scores.append(1.)

                            # print('y: %s' % ' '.join(new_tgt_sent))
                            tgt_samples.append(new_tgt_sent)

                        # if enable importance sampling, compute importance weight
                        if args.raml_sample_mode == 'hamming_distance_impt_sample':
                            tgt_sample_weights = [math.exp(bleu_score / tau) / math.exp(-e / tau) for e, bleu_score in
                                                  zip(e_samples, bleu_scores)]
                            normalizer = sum(tgt_sample_weights)
                            tgt_sample_weights = [w / normalizer for w in tgt_sample_weights]
                        else:
                            tgt_sample_weights = [1.] * args.sample_size

                        raml_src_sents.extend([src_sent] * len(tgt_samples))
                        raml_tgt_sents.extend(tgt_samples)
                        raml_tgt_weights.extend(tgt_sample_weights)

                        if args.debug:
                            print('*' * 30)
                            print('Target: %s' % ' '.join(tgt_sent))
                            for tgt_sample, e, bleu_score, weight in zip(tgt_samples, e_samples, bleu_scores,
                                                                         tgt_sample_weights):
                                print('Sample: %s ||| e: %d ||| bleu: %f ||| weight: %f' % (
                                    ' '.join(tgt_sample), e, bleu_score, weight))
                            print()
                            break

                src_sents_var = to_input_variable(raml_src_sents, vocab.src, cuda=args.cuda)
                tgt_sents_var = to_input_variable(raml_tgt_sents, vocab.tgt, cuda=args.cuda)
                weights_var = Variable(torch.FloatTensor(raml_tgt_weights), requires_grad=False)
                if args.cuda:
                    weights_var = weights_var.cuda()

                batch_size = len(raml_src_sents)  # batch_size = args.batch_size * args.sample_size
                src_sents_len = [len(s) for s in raml_src_sents]
                pred_tgt_word_num = sum(len(s[1:]) for s in raml_tgt_sents)  # omitting leading `<s>`
                optimizer.zero_grad()

                # (tgt_sent_len, batch_size, tgt_vocab_size)
                scores = model(src_sents_var, src_sents_len, tgt_sents_var[:-1])
                # (tgt_sent_len * batch_size, tgt_vocab_size)
                log_scores = F.log_softmax(scores.view(-1, scores.size(2)))
                # remove leading <s> in tgt sent, which is not used as the target
                flattened_tgt_sents = tgt_sents_var[1:].view(-1)

                # batch_size * tgt_sent_len
                tgt_log_scores = torch.gather(log_scores, 1, flattened_tgt_sents.unsqueeze(1)).squeeze(1)
                unweighted_loss = -tgt_log_scores * (1. - torch.eq(flattened_tgt_sents, 0).float())
                weighted_loss = unweighted_loss * weights_var.repeat(scores.size(0))
                weighted_loss = weighted_loss.sum()
                weighted_loss_val = weighted_loss.item()
                nll_loss_val = unweighted_loss.sum().item()
                # weighted_log_scores = log_scores * weights.view(-1, scores.size(2))
                # weighted_loss = nll_loss(weighted_log_scores, flattened_tgt_sents)

                loss = weighted_loss / batch_size
                # nll_loss_val = nll_loss(log_scores, flattened_tgt_sents).item()

                loss.backward()
                # clip gradient
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

                report_weighted_loss += weighted_loss_val
                cum_weighted_loss += weighted_loss_val
                report_loss += nll_loss_val
                cum_loss += nll_loss_val
                report_tgt_words += pred_tgt_word_num
                cum_tgt_words += pred_tgt_word_num
                report_examples += batch_size
                cum_examples += batch_size
                cum_batches += batch_size

                if train_iter % args.log_every == 0:
                    _log = 'epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (
                        epoch, train_iter,
                        report_weighted_loss / report_examples,
                        np.exp(report_loss / report_tgt_words),
                        cum_examples,
                        report_tgt_words / (time.time() - train_time),
                        time.time() - begin_time)
                    print(_log)
                    print(_log, file=train_output)
                    train_time = time.time()
                    report_loss = report_weighted_loss = report_tgt_words = report_examples = 0.
                    if args.notify_slack:
                        slack.post(_log)

                # perform validation
                if train_iter % args.valid_niter == 0:
                    print('epoch %d, iter %d, cum. loss %.2f, '
                          'cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                              cum_weighted_loss / cum_batches,
                                                              np.exp(cum_loss / cum_tgt_words),
                                                              cum_examples),
                          file=sys.stderr)

                    cum_loss = cum_weighted_loss = cum_batches = cum_tgt_words = 0.
                    valid_num += 1

                    print('begin validation ...')
                    model.eval()

                    # compute dev. ppl and bleu

                    dev_loss = evaluate_loss(model, dev_data, cross_entropy_loss)
                    dev_ppl = np.exp(dev_loss)

                    if args.valid_metric in ['bleu', 'word_acc', 'sent_acc']:
                        dev_hyps = decode(model, dev_data, f=validation_output, verbose=False)
                        dev_hyps = [hyps[0] for hyps in dev_hyps]
                        if args.valid_metric == 'bleu':
                            valid_metric = get_bleu([tgt for src, tgt in dev_data], dev_hyps)
                        else:
                            valid_metric = get_acc([tgt for src, tgt in dev_data], dev_hyps, acc_type=args.valid_metric)
                        _log = 'validation: iter %d, dev. ppl %f, dev. %s %f' % (
                            train_iter, dev_ppl, args.valid_metric, valid_metric)
                        print(_log)
                        print(_log, file=validation_output)
                        if args.notify_slack:
                            slack.post(_log)

                    else:
                        valid_metric = -dev_ppl
                        print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl),
                              file=sys.stderr)

                    model.train()

                    is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                    is_better_than_last = len(hist_valid_scores) == 0 or valid_metric > hist_valid_scores[-1]
                    hist_valid_scores.append(valid_metric)

                    if valid_num > args.save_model_after:
                        model_file = args.save_to + '.iter%d.bin' % train_iter
                        print('save model to [%s]' % model_file)
                        model.save(model_file)

                    if (not is_better_than_last) and args.lr_decay:
                        lr = optimizer.param_groups[0]['lr'] * args.lr_decay
                        print('decay learning rate to %f' % lr)
                        optimizer.param_groups[0]['lr'] = lr

                    if is_better:
                        patience = 0
                        best_model_iter = train_iter

                        if valid_num > args.save_model_after:
                            print('save currently the best model ..')
                            model_file_abs_path = os.path.abspath(model_file)
                            symlin_file_abs_path = os.path.abspath(args.save_to + '.bin')
                            os.system('ln -sf %s %s' % (model_file_abs_path, symlin_file_abs_path))
                    else:
                        patience += 1
                        print('hit patience %d' % patience)
                        if patience == args.patience:
                            _log = f"""
                            {'hit patience %d' % patience}
                            early stop!
                            {'the best model is from iteration [%d]' % best_model_iter}
                            """
                            print(_log)
                            if args.notify_slack:
                                slack.post(_log)
                            exit(0)

                if args.debug:
                    print(f'debug epoch:{epoch} exit')
                    model_file = args.save_to + '.bin'
                    print('save model to [%s]' % model_file)
                    model.save(model_file)
                    exit(0)


def get_bleu(references, hypotheses):
    # compute BLEU
    bleu_score = corpus_bleu([[ref[1:-1]] for ref in references],
                             [hyp[1:-1] for hyp in hypotheses])

    return bleu_score


def get_acc(references, hypotheses, acc_type='word'):
    assert acc_type == 'word_acc' or acc_type == 'sent_acc'
    cum_acc = 0.

    for ref, hyp in zip(references, hypotheses):
        ref = ref[1:-1]
        hyp = hyp[1:-1]
        if acc_type == 'word_acc':
            acc = len([1 for ref_w, hyp_w in zip(ref, hyp) if ref_w == hyp_w]) / float(len(hyp) + 1e-6)
        else:
            acc = 1. if all(ref_w == hyp_w for ref_w, hyp_w in zip(ref, hyp)) else 0.
        cum_acc += acc

    acc = cum_acc / len(hypotheses)
    return acc


def decode(model, data, verbose=True, f=None):
    """
    decode the dataset and compute sentence level acc. and BLEU.
    """
    hypotheses = []
    begin_time = time.time()

    print('decode %d examples' % (len(data)))

    if type(data[0]) is tuple:
        for src_sent, tgt_sent in data:
            hyps = model.translate(src_sent)
            hypotheses.append(hyps)

            if verbose or f:
                report = f"""
                {'*' * 50}
                Source: {' '.join(src_sent)}
                Target: {' '.join(tgt_sent)}
                Top Hypothesis: {' '.join(hyps[0])}
                """
                if verbose: print(report);
                if f: print(report, file=f);
    else:
        for src_sent in data:
            hyps = model.translate(src_sent)
            hypotheses.append(hyps)

            if verbose or f:
                report = f"""
                {'*' * 50}
                Source: {' '.join(src_sent)}
                Top Hypothesis: {' '.join(hyps[0])}
                """
                if verbose: print(report);
                if f: print(report, file=f);

    elapsed = time.time() - begin_time

    print('decoded %d examples, took %d s' % (len(data), elapsed))

    return hypotheses


def compute_lm_prob(args):
    """
    given source-target sentence pairs, compute ppl and log-likelihood
    """
    test_data_src = read_corpus(args.test_src, source='src')
    test_data_tgt = read_corpus(args.test_tgt, source='tgt')
    test_data = zip(test_data_src, test_data_tgt)

    if args.load_model:
        print('load model from [%s]' % args.load_model)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        state_dict = params['state_dict']

        model = NMT(saved_args, vocab)
        model.load_state_dict(state_dict)
    else:
        vocab = torch.load(args.vocab)
        model = NMT(args, vocab)

    model.eval()

    if args.cuda:
        model = nn.DataParallel(model).cuda()

    f = open(args.save_to_file, 'w')
    for src_sent, tgt_sent in test_data:
        src_sents = [src_sent]
        tgt_sents = [tgt_sent]

        batch_size = len(src_sents)
        src_sents_len = [len(s) for s in src_sents]
        pred_tgt_word_nums = [len(s[1:]) for s in tgt_sents]  # omitting leading `<s>`

        # (sent_len, batch_size)
        src_sents_var = to_input_variable(src_sents, model.vocab.src, cuda=args.cuda, is_test=True)
        tgt_sents_var = to_input_variable(tgt_sents, model.vocab.tgt, cuda=args.cuda, is_test=True)

        # (tgt_sent_len, batch_size, tgt_vocab_size)
        scores = model(src_sents_var, src_sents_len, tgt_sents_var[:-1])
        # (tgt_sent_len * batch_size, tgt_vocab_size)
        log_scores = F.log_softmax(scores.view(-1, scores.size(2)))
        # remove leading <s> in tgt sent, which is not used as the target
        # (batch_size * tgt_sent_len)
        flattened_tgt_sents = tgt_sents_var[1:].view(-1)
        # (batch_size * tgt_sent_len)
        tgt_log_scores = torch.gather(log_scores, 1, flattened_tgt_sents.unsqueeze(1)).squeeze(1)
        # 0-index is the <pad> symbol
        tgt_log_scores = tgt_log_scores * (1. - torch.eq(flattened_tgt_sents, 0).float())
        # (tgt_sent_len, batch_size)
        tgt_log_scores = tgt_log_scores.view(-1, batch_size)  # .permute(1, 0)
        # (batch_size)
        tgt_sent_scores = tgt_log_scores.sum(dim=0).squeeze()
        tgt_sent_word_scores = [tgt_sent_scores[i].item() / pred_tgt_word_nums[i] for i in range(batch_size)]

        for src_sent, tgt_sent, score in zip(src_sents, tgt_sents, tgt_sent_word_scores):
            f.write('%s ||| %s ||| %f\n' % (' '.join(src_sent), ' '.join(tgt_sent), score))

    f.close()


def test(args):
    test_data_src = read_corpus(args.test_src, source='src')
    test_data_tgt = read_corpus(args.test_tgt, source='tgt')
    test_data = list(zip(test_data_src, test_data_tgt))

    if args.load_model:
        print('load model from [%s]' % args.load_model)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        state_dict = params['state_dict']

        model = NMT(saved_args, vocab)
        model.load_state_dict(state_dict)
    else:
        vocab = torch.load(args.vocab)
        model = NMT(args, vocab)

    model.eval()

    if args.cuda:
        # model = nn.DataParallel(model).cuda()
        model = model.cuda()

    hypotheses = decode(model, test_data)
    top_hypotheses = [hyps[0] for hyps in hypotheses]

    bleu_score = get_bleu([tgt for src, tgt in test_data], top_hypotheses)
    word_acc = get_acc([tgt for src, tgt in test_data], top_hypotheses, 'word_acc')
    sent_acc = get_acc([tgt for src, tgt in test_data], top_hypotheses, 'sent_acc')
    print('Corpus Level BLEU: %f, word level acc: %f, sentence level acc: %f' % (bleu_score, word_acc, sent_acc),
          file=sys.stderr)

    if args.save_to_file:
        print('save decoding results to %s' % args.save_to_file)
        with open(args.save_to_file, 'w') as f:
            for hyps in hypotheses:
                f.write(' '.join(hyps[0][1:-1]) + '\n')

        if args.save_nbest:
            nbest_file = args.save_to_file + '.nbest'
            print('save nbest decoding results to %s' % nbest_file)
            with open(nbest_file, 'w') as f:
                for src_sent, tgt_sent, hyps in zip(test_data_src, test_data_tgt, hypotheses):
                    print('Source: %s' % ' '.join(src_sent), file=f)
                    print('Target: %s' % ' '.join(tgt_sent), file=f)
                    print('Hypotheses:', file=f)
                    for i, hyp in enumerate(hyps, 1):
                        print('[%d] %s' % (i, ' '.join(hyp)), file=f)
                    print('*' * 30, file=f)


def interactive(args):
    assert args.load_model, 'You have to specify a pre-trained model'
    print('load model from [%s]' % args.load_model)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    vocab = params['vocab']
    saved_args = params['args']
    state_dict = params['state_dict']

    model = NMT(saved_args, vocab)
    model.load_state_dict(state_dict)

    model.eval()

    if args.cuda:
        model = nn.DataParallel(model).cuda()

    while True:
        src_sent = input('Source Sentence:')
        src_sent = src_sent.strip().split(' ')
        hyps = model.translate(src_sent)
        for i, hyp in enumerate(hyps, 1):
            print('Hypothesis #%d: %s' % (i, ' '.join(hyp)))


def sample(args):
    train_data_src = read_corpus(args.train_src, source='src')
    train_data_tgt = read_corpus(args.train_tgt, source='tgt')
    train_data = zip(train_data_src, train_data_tgt)

    if args.load_model:
        print('load model from [%s]' % args.load_model)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        opt = params['args']
        state_dict = params['state_dict']

        model = NMT(opt, vocab)
        model.load_state_dict(state_dict)
    else:
        vocab = torch.load(args.vocab)
        model = NMT(args, vocab)

    model.eval()

    if args.cuda:
        model = nn.DataParallel(model).cuda()

    print('begin sampling')

    check_every = 10
    train_iter = cum_samples = 0
    train_time = time.time()
    for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
        train_iter += 1
        samples = model.sample(src_sents, sample_size=args.sample_size, to_word=True)
        cum_samples += sum(len(sample) for sample in samples)

        if train_iter % check_every == 0:
            elapsed = time.time() - train_time
            print('sampling speed: %d/s' % (cum_samples / elapsed))
            cum_samples = 0
            train_time = time.time()

        for i, tgt_sent in enumerate(tgt_sents):
            print('*' * 80)
            print('target:' + ' '.join(tgt_sent))
            tgt_samples = samples[i]
            print('samples:')
            for sid, sample in enumerate(tgt_samples, 1):
                print('[%d] %s' % (sid, ' '.join(sample[1:-1])))
            print('*' * 80)


if __name__ == '__main__':
    args = init_config()
    print(args)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'raml_train':
        train_raml(args)
    elif args.mode == 'sample':
        sample(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'prob':
        compute_lm_prob(args)
    elif args.mode == 'interactive':
        interactive(args)
    else:
        raise RuntimeError('unknown mode')
