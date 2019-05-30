# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:14:46 2016

@author: Balázs Hidasi
"""

import sys
import argparse
sys.path.append('../..')

import numpy as np
import pandas as pd
import gru4rec
import evaluation

#PATH_TO_TRAIN = '/db_vol/hb_work/rnn/data/processed/recsys_challenge_train_full.txt'
#PATH_TO_TEST = '/db_vol/hb_work/rnn/data/processed/recsys_challenge_test.txt'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Example with long option names',
    )

    parser.add_argument('--train', action="store",
                        dest="train")
    parser.add_argument('--test', action="store",
                        dest="test")
    parser.add_argument('--hidden', action="store", dest='hidden', default=100, type=int)
    parser.add_argument('--epoch', action="store", dest='epoch', default=1, type=int)
    parser.add_argument('--batch_size', action="store", dest='batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', action="store", dest='learning_rate', default=0.1, type=float)
    parser.add_argument('--n_sample', action="store", dest='n_sample', default=2048, type=int)
    parser.add_argument('--embedding_file', action="store", dest='embedding_file', default=None)
    parser.add_argument('--adapt', action="store", dest='adapt', default=None)
    args=parser.parse_args()

    PATH_TO_TRAIN=args.train
    PATH_TO_TEST=args.test
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId':np.int64})
    valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId':np.int64})
    
    #State-of-the-art results on RSC15 from "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations" on RSC15 (http://arxiv.org/abs/1706.03847)
    #BPR-max, no embedding (R@20 = 0.7197, M@20 = 0.3157)
    #gru = gru4rec.GRU4Rec(loss='bpr-max', final_act='elu-0.5', hidden_act='tanh', layers=[100], adapt='adagrad', n_epochs=10, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0, learning_rate=0.2, momentum=0.3, n_sample=2048, sample_alpha=0, bpreg=1, constrained_embedding=False)
    #gru.fit(data)
    #res = evaluation.evaluate_gpu(gru, valid)
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1]))

    #BPR-max, constrained embedding (R@20 = 0.7261, M@20 = 0.3124)
    #gru = gru4rec.GRU4Rec(loss='bpr-max', final_act='elu-0.5', hidden_act='tanh', layers=[100], adapt='adagrad', n_epochs=10, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0, learning_rate=0.2, momentum=0.1, n_sample=2048, sample_alpha=0, bpreg=0.5, constrained_embedding=True)
    #gru.fit(data)
    #res = evaluation.evaluate_gpu(gru, valid)
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1]))

    #Cross-entropy (R@20 = 0.7180, M@20 = 0.3087)
    #gru = gru4rec.GRU4Rec(loss='cross-entropy', final_act='softmax', hidden_act='tanh', layers=[100], adapt='adagrad', n_epochs=10, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0.3, learning_rate=0.1, momentum=0.7, n_sample=2048, sample_alpha=0, bpreg=0, constrained_embedding=False)
    if args.embedding_file is None:
        emb = 0
        E = np.array([0])
    else:
        emb = -1
        f=open(args.embedding_file, 'rb')
        E = np.load(f)
        f.close()

    gru = gru4rec.GRU4Rec(
        loss='cross-entropy', 
        final_act='softmax', 
        hidden_act='tanh', 
        layers=[args.hidden], 
        adapt='adagrad', 
        n_epochs=args.epoch, 
        batch_size=args.batch_size, 
        dropout_p_embed=0, 
        dropout_p_hidden=0.3, 
        learning_rate=args.learning_rate, 
        momentum=0.7, 
        n_sample=args.n_sample, 
        sample_alpha=0, 
        bpreg=0, 
        constrained_embedding=False,
        embedding=emb,
        embedding_matrix=E
        )
    gru.fit(data)
    res = evaluation.evaluate_gpu(gru, valid)
    print('Recall@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))

    #TOP1
    # gru = gru4rec.GRU4Rec(
    #     loss='top1', 
    #     final_act='softmax', 
    #     hidden_act='tanh', 
    #     layers=[args.hidden], 
    #     adapt='adagrad', 
    #     n_epochs=args.epoch, 
    #     batch_size=args.batch_size, 
    #     dropout_p_embed=0, 
    #     dropout_p_hidden=0.3, 
    #     learning_rate=args.learning_rate, 
    #     momentum=0.7, 
    #     n_sample=args.n_sample, 
    #     sample_alpha=0, 
    #     bpreg=0, 
    #     constrained_embedding=False)
    # gru.fit(data)
    # res = evaluation.evaluate_gpu(gru, valid)
    # print('Recall@20: {}'.format(res[0]))
    # print('MRR@20: {}'.format(res[1]))
    
    #OUTDATED!!!
    #Reproducing results from the original paperr"Session-based Recommendations with Recurrent Neural Networks" on RSC15 (http://arxiv.org/abs/1511.06939)
    #print('Training GRU4Rec with 100 hidden units')    
    #gru = gru4rec.GRU4Rec(loss='top1', final_act='tanh', hidden_act='tanh', layers=[100], batch_size=50, dropout_p_hidden=0.5, learning_rate=0.01, momentum=0.0, time_sort=False)
    #gru.fit(data)
    #res = evaluation.evaluate_gpu(gru, valid)
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1]))
