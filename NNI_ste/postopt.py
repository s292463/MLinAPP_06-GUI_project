import os
import numpy as np
import pandas as pd
import json
import random as rn
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.signal import butter, freqz
import nengo
import nengo_dl
from nengo.utils.filter_design import cont2discrete
import tensorflow as tf
import keras
from keras.callbacks import Callback
from keras.models import Sequential
from keras.models import Model
from keras.layers import *
from keras.utils import to_categorical
from keras.regularizers import l2,l1
from keras.optimizers import Adam
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph



def load_wisdm2_data(filename):
    filepath = os.path.join('./data/',filename+'.npz')
    a = np.load(filepath)
    return (a['arr_0'], a['arr_1'], a['arr_2'], a['arr_3'], a['arr_4'], a['arr_5'])


def load_dataset(file_name):
    filepath = os.path.join('./data/',file_name+'.npz')
    a = np.load(filepath)
    return (a['arr_0'], a['arr_1'], a['arr_2'], a['arr_3'], a['arr_4'], a['arr_5'])


class DeviceData:
    def __init__(self, sample, fs, channels):
        self.data = []
        sample = sample.T
        for data_axis in range(sample.shape[0]):
            self.data.append(sample[data_axis, :])

        self.fs = fs
        self.freq_range = (0.5, np.floor(self.fs / 2))

        freq_min, freq_max = self.freq_range
        octave = (channels - 0.5) * np.log10(2) / np.log10(freq_max / freq_min)
        self.freq_centr = np.array([freq_min * (2 ** (ch / octave)) for ch in range(channels)])
        self.freq_poli = np.array(
            [(freq * (2 ** (-1 / (2 * octave))), (freq * (2 ** (1 / (2 * octave))))) for freq in self.freq_centr])
        self.freq_poli[-1, 1] = fs / 2 * 0.99999

    def decomposition(self, filterbank):
        self.components = []
        for data_axis in self.data:
            tmp = []
            for num, den in filterbank:
                from scipy.signal import lfilter
                tmp.append(lfilter(num, den, data_axis))
            self.components.append(tmp)


def frequency_decomposition(array, channels=5, fs=20, order=2):

    array_dec = []

    for ii in range(len(array)):
    
        sample = DeviceData(array[ii], fs, channels)
    
        butter_filterbank = []
        for fl, fh in sample.freq_poli:
            num, den = butter(N=order, Wn=(fl, fh), btype='band', fs=sample.fs)
            butter_filterbank.append([num, den])
    
        sample.decomposition(butter_filterbank)
    
        features = []
        for data_axis in sample.components:
            for component in data_axis:
                features.append(np.array(component))
        features = np.vstack(features)
        features = features.T
    
        array_dec.append(features)

    return np.array(array_dec)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend(['train_'+string, 'val_'+string])
    plt.show()


def ConfusionMatrix_labels(y_train, y_test):
    
    ys = np.concatenate((y_train,y_test))
    
    if (type(ys[0]) == list) | (type(ys[0]) == np.ndarray):
        if len(ys[0]) > 1:
            ys = np.argmax(ys,axis=1)
            labels = np.unique(ys).tolist()
        else:
            labels = np.unique(ys).tolist()
    else:
        labels = np.unique(np.array(ys)).tolist()
    
    return labels


def ConfusionMatrix_wisdm2_labels(subset=2):
    
    act_map = {
        'A': 'walking',
        'B': 'jogging',
        'C': 'stairs',
        'D': 'sitting',
        'E': 'standing',
        'M': 'kicking',
        'P': 'dribbling',
        'O': 'catch',
        'F': 'typing',
        'Q': 'writing',
        'R': 'clapping',
        'G': 'teeth',
        'S': 'folding',
        'J': 'pasta',
        'H': 'soup',
        'L': 'sandwich',
        'I': 'chips',
        'K': 'drinking',
    }
    
    if subset == 1:
        labels = list(act_map.values())[:6]
    if subset == 2:
        labels = list(act_map.values())[6:13]
    if subset == 3:
        labels = list(act_map.values())[13:]
    
    return labels


def memory_footprint(model, nengo=True):
    
    mem_fp = 0
    total = 0
    missed = 0
    
    if nengo:
        
        model_weights = model.keras_model.weights
    
    else:
        
        model_weights = model.weights
        
    for s in model_weights:
        if ('32' in str(s.dtype)) or (str(s.dtype)=='int'):
            mem = 4*np.prod(s.shape)/1e6 # MB
            total += np.prod(s.shape)
            mem_fp += mem
        elif '64' in str(s.dtype):
            mem = 8*np.prod(s.shape)/1e6 # MB
            total += np.prod(s.shape)
            mem_fp += mem
        else:
            missed += np.prod(s.shape)
    
    return mem_fp, total, missed


def get_flops(model):
    
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function([tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        
        return flops.total_float_ops


class LMUCell(nengo.Network):
    def __init__(self, units, order, theta, input_d, tau, **kwargs):
        super().__init__(**kwargs)

        Q = np.arange(order, dtype=np.float64)
        R = (2 * Q + 1)[:, None] / theta
        j, i = np.meshgrid(Q, Q)

        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R 
        B = (-1.0) ** Q[:, None] * R 
        C = np.ones((1, order))
        D = np.zeros((1,))

        A, B, _, _, _ = cont2discrete((A, B, C, D), dt=tau, method="zoh") # original: dt=1.0

        A_H = 1/(1-np.exp(-1/tau)) * (A - np.exp(-1/tau)*np.identity(order))
        B_H = 1/(1-np.exp(-1/tau)) * B


        with self:
            nengo_dl.configure_settings(trainable=None)

            # create objects corresponding to the x/u/m/h
            self.x = nengo.Node(size_in=input_d)
            self.u = nengo.Node(size_in=1)
            self.m = nengo.Node(size_in=order)
            self.h = nengo_dl.TensorNode(tf.nn.tanh, shape_in=(units,), pass_time=False)

            # compute u_t:
            # e_x
            nengo.Connection(
                self.x, self.u, transform=np.ones((1, input_d)), synapse=None
            )
            
            # e_h
            nengo.Connection(
                self.h, self.u, transform=np.ones((1, units)), synapse=0
            )
            
            # e_m
            nengo.Connection(
                self.m, self.u, transform=np.ones((1, order)), synapse=0
            )

            # compute m_t:
            conn_A = nengo.Connection(self.m, self.m, transform=A_H, synapse=0)
            self.config[conn_A].trainable = True
            conn_B = nengo.Connection(self.u, self.m, transform=B_H, synapse=None)
            self.config[conn_B].trainable = True

            # compute h_t:
            nengo.Connection(
                self.x, self.h, transform=nengo_dl.dists.Glorot(), synapse=None
            )
            nengo.Connection(
                self.h, self.h, transform=nengo_dl.dists.Glorot(), synapse=0
            )
            nengo.Connection(
                self.m, self.h, transform=nengo_dl.dists.Glorot(), synapse=None,
            )
