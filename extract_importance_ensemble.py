#!/bin/env python
import os
import numpy as np 
import argparse
from ensemble_utils import *
from CNN import *
import argparse
import keras
import pickle
from tensorflow.python.client import device_lib
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import activations
from importance_utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print(device_lib.list_local_devices())

parser = argparse.ArgumentParser()
parser.add_argument('testfasta')
parser.add_argument('comparisons')
parser.add_argument('model',help="model folder")
parser.add_argument('outfile')
opts=parser.parse_args()

X = fa_to_onehot(opts.testfasta)
model_folders = [opts.model+"/"+d for d in os.listdir(opts.model) if os.path.isdir(opts.model+"/"+d)]
with open(opts.model+"/model_acc.pkl","rb") as f:
    accuracies = pickle.load(f)
    accuracies = {key.split('/')[-1]:accuracies[key] for key in accuracies.keys()}

comps = [tuple(l.strip().split()) for l in open(opts.comparisons)]
print(comps)
for comp in comps:
    print(comp)
    if comp[0] == 'None':
        c1 = []
        c2 = [int(c) for c in comp[1].split('-')]
    elif comp[1] == 'None':
        c2 = []
        c1 = [int(c) for c in comp[0].split('-')]
    else:
        c1 = [int(c) for c in comp[0].split('-')]
        c2 = [int(c) for c in comp[1].split('-')]

    for mi,model in enumerate(model_folders):
        if mi  == 0:
            grads_i = accuracies[model.split('/')[-1]]*saliency(0,model+"/model.h5",X,c1,c2)*X
        else:
            grads_i += accuracies[model.split('/')[-1]]*saliency(0,model+"/model.h5",X,c1,c2)*X
        
    # grads are a X size matrix with importance scores for each
    # sequence, for each position in the sequence
    with open(opts.outfile+'_'+comp[0]+'vs'+comp[1]+'.pkl', 'wb') as handle:
        pickle.dump(grads_i, handle, protocol=2)
        




