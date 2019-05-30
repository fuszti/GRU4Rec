# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:20:12 2015

@author: Balázs Hidasi
"""

import argparse
import numpy as np
import pandas as pd
import datetime as dt
import os
import random

#PATH_TO_ORIGINAL_DATA = '/path/to/clicks/dat/file/'
#PATH_TO_PROCESSED_DATA = '/path/to/store/processed/data/'
PATH_TO_ORIGINAL_DATA = 'D:\\FusztiG\\Elte\\MSc\\Szakdolgozat\\data\\processed\\' #yoochoose-data\\'
PATH_TO_PROCESSED_DATA = 'D:\\FusztiG\\Elte\\MSc\\Szakdolgozat\\data\\processed\\'
random.seed(42)
parser = argparse.ArgumentParser(
    description='Example with long option names',
)

parser.add_argument('--input', action="store",
                    dest="inp")
parser.add_argument('--output', action="store",
                    dest="outp")
args=parser.parse_args()

PATH_TO_ORIGINAL_DATA=args.inp
PATH_TO_PROCESSED_DATA=args.outp

#data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'yoochoose-clicks.dat', sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64})
#data.columns = ['SessionId', 'TimeStr', 'ItemId']
#data['Time'] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) #This is not UTC. It does not really matter.
#del(data['TimeStr'])

data=pd.read_csv(PATH_TO_ORIGINAL_DATA, sep=',', usecols=['time', 'user', 'item'], dtype={'time':np.int64, 'user':np.int64, 'item':np.int64})
data=data.rename(index=str, columns={"time": "Time", "user": "SessionId", "item":"ItemId"})

session_lengths = data.groupby('SessionId').size()
data = data[np.in1d(data.SessionId, session_lengths[session_lengths>1].index)]
item_supports = data.groupby('ItemId').size()
data = data[np.in1d(data.ItemId, item_supports[item_supports>=5].index)]
session_lengths = data.groupby('SessionId').size()
data = data[np.in1d(data.SessionId, session_lengths[session_lengths>=2].index)]
session_lengths = data.groupby('SessionId').size()
print(session_lengths)
#print('session_lengths = {}\nitem_supports = {}\n'.format(session_lengths, item_supports))

tmax = data.Time.max()
tmin = data.Time.min()
print('tmax={}\ntmin={}\ntmax-tmin= {}'.format(tmax, tmin,tmax-tmin))
session_max_times = data.groupby('SessionId').Time.max()
split_value = int((tmax-tmin)*0.3)
session_index_list = session_lengths.index.tolist()
random.shuffle(session_index_list)
#session_prob_list = session_lengths.values.tolist()
K = int(len(session_index_list)*0.7)
session_train = session_index_list[0:K]
session_test = session_index_list[K:]
#session_train = random.choices(session_index_list, weights=session_prob_list, k=K)
print('-------------------------')
print(session_max_times.index)
print(session_max_times[[5, 9, 11]].index)
print(session_lengths[[5, 9, 11]].values)
#print(session_max_times[-1])
print('-------------------------')
#session_train = session_max_times[session_max_times < tmax-split_value].index
#session_test = session_max_times[session_max_times >= tmax-split_value].index
train = data[np.in1d(data.SessionId, session_train)]
test = data[np.in1d(data.SessionId, session_test)]
test = test[np.in1d(test.ItemId, train.ItemId)]
tslength = test.groupby('SessionId').size()
test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
train.to_csv(os.path.join(PATH_TO_PROCESSED_DATA, 'rsc15_train_full.txt'), sep='\t', index=False)
print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
test.to_csv(os.path.join(PATH_TO_PROCESSED_DATA,'rsc15_test.txt'), sep='\t', index=False)

tmax = train.Time.max()
tmin = train.Time.min()
print('tmax={}\ntmin={}\ntmax-tmin= {}'.format(tmax, tmin,tmax-tmin))
session_max_times = train.groupby('SessionId').Time.max()
split_value = int((tmax-tmin)*0.3)

session_lengths = train.groupby('SessionId').size()
session_index_list = session_lengths.index.tolist()
random.shuffle(session_index_list)
#session_prob_list = session_lengths.values.tolist()
K = int(len(session_index_list)*0.7)
session_train = session_index_list[0:K]
session_valid = session_index_list[K:]

#session_train = session_max_times[session_max_times < tmax-split_value].index
#session_valid = session_max_times[session_max_times >= tmax-split_value].index
train_tr = train[np.in1d(train.SessionId, session_train)]
valid = train[np.in1d(train.SessionId, session_valid)]
valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
tslength = valid.groupby('SessionId').size()
valid = valid[np.in1d(valid.SessionId, tslength[tslength>=2].index)]
print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))
train_tr.to_csv(os.path.join(PATH_TO_PROCESSED_DATA, 'rsc15_train_tr.txt'), sep='\t', index=False)
print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
valid.to_csv(os.path.join(PATH_TO_PROCESSED_DATA, 'rsc15_train_valid.txt'), sep='\t', index=False)
