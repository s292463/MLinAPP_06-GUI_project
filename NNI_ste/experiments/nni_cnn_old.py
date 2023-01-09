import pprint
from nni.experiment.config.base import yaml

import tensorflow as tf
from keras.models import Sequential, Model
from keras import Input
from keras.layers import  Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop
from keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
import argparse
import logging

from sklearn.model_selection import train_test_split

import nni
import numpy as np
import os
import shutil
import json
from nni.tuner import Tuner
from nni.experiment import Experiment
from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
from nni.tools.nnictl import updater, nnictl_utils
import csv
# import for resampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from tensorflow.python.ops.array_ops import inplace_sub_eager_fallback
from tensorflow.python.ops.nn_ops import conv3d_backprop_filter_v2


os.environ["CUDA_VISIBLE_DEVICES"] = '0'  
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'  


seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
rng = np.random.RandomState(seed)


### SET NETWORK TYPE:
network_type = 'cnn'

##### DATASET-RELATED SETTINGS #######################

# name to identify experiments on different datasets
superclass = 'superclass2'
dataset_name = 'miRNA-'+superclass
folder_name= '../data-knn'
dropout_flag = False

# file name of the dataset
data_file = '../'+folder_name+'/'+superclass+'/data.csv'
label_file = '../'+folder_name+'/'+superclass+'/labels.csv'

######################################################

if dropout_flag == False:
  searchspace_path = '..searchspaces/nni_SearchSpace_'+network_type+'.json'
else:
  searchspace_path = '..searchspaces/nni_SearchSpace_'+network_type+'_DO.json'

print(searchspace_path)
eps = 100

def count_single_label(x, label):
  count = 0
  for idx, l in enumerate(x):
    if l == label:
      count = count+1
  if count <= 3:
    return True
  else:
    return False

def load_dataset(data_file_name, label_file_name):
    miRna_label = extract_label(label_file_name)
    miRna_data = np.genfromtxt(data_file_name, delimiter=',')
    return miRna_data, miRna_label

def extract_label(file_name):
    label = []
    with open(file_name, "r") as fin:
        reader = csv.reader(fin, delimiter=',')
        first = True
        for row in reader:
            label.append(row[0])
    return np.array(label)

def create_dictionary(labels):
    dictionary = {}
    class_names = np.unique(labels)
    for i, name in enumerate(class_names):
        dictionary[name] = i
    return dictionary

def label_processing2(labels):
    new_miRna_label = []
    count = {}
    dictionary = create_dictionary(labels)
    for label in labels:
      new_miRna_label.append(dictionary[label])
    return  np.array(new_miRna_label)

import operator
def duplicate_single_values(x,y):

  # Duplication in order to have potentially one sample per train, test, validation
  print("#######################\n\t\t Data Analysis for Stratify...")
  count = {}
  data = list(x)
  labels = list(y)
  print("Before duplication:", y.shape[0])
  for label in np.unique(y):
   count[label] = 0
  for label in y:
    count[label] = count[label] + 1
  print(sorted(count.items(), key=operator.itemgetter(1)))
  for k,v in count.items():
    if(v < 3):
      print(f"{k} = {v}")
  for idx,label in enumerate(y):
    if(count[label] < 3):
      data.append(x[idx, :])
      labels.append(y[idx])
      count[label] = count[label] + 1
  data, labels = np.array(data), np.array(labels)
  print("After:", labels.shape[0])
  print("#######################")
  return data,labels



def report_result(args, result, result_type):

    if result_type == 'test':
        report_file = out_dir + 'nni_' + args.network_type + '_' + nni.get_experiment_id() + '_' + result_type + '_accs'
    else:
        report_file = out_dir + 'nni_' + args.network_type + '_' + nni.get_experiment_id() + '_' + result_type + '_accs_' + nni.get_trial_id()
    
    with open(report_file, 'a') as f:
        f.write(str(result))
        f.write('\n')
    
    return report_file


class SendMetrics(Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
        #LOG.debug(logs)
        
        if 'val_acc' in logs:
            nni.report_intermediate_result(logs['val_acc'])
        else:
            nni.report_intermediate_result(logs['val_accuracy'])


def create_MLinApp_model_dropout(args, hyper_params, timesteps, input_dim, n_classes):
    inp = tf.keras.Input(shape=(input_dim, timesteps))
    conv0 = Conv1D(filters=hyper_params['nni_network/Conv1D_filters_1/randint'], kernel_size=hyper_params['nni_network/Conv1D_kernel_size_1/randint'], activation=tf.nn.relu, kernel_initializer='he_uniform', input_shape=(input_dim, timesteps), name='Conv1D_1')(inp)
    maxPool0 = MaxPooling1D(pool_size=hyper_params['nni_network/MaxPooling1D_kernel_size_1/randint'], name='MaxPooling1D_1')(conv0)

    conv1 = Conv1D(filters=hyper_params['nni_network/Conv1D_filters_2/randint'], kernel_size=hyper_params['nni_network/Conv1D_kernel_size_2/randint'], kernel_initializer='he_uniform', activation=tf.nn.relu, name='Conv1D_2')(maxPool0)
    maxPool1 = MaxPooling1D(pool_size=hyper_params['nni_network/MaxPooling1D_kernel_size_2/randint'], name='MaxPooling1D_2')(conv1)
    
    conv2 = Conv1D(filters=hyper_params['nni_network/Conv1D_filters_3/randint'], kernel_size=hyper_params['nni_network/Conv1D_kernel_size_3/randint'], kernel_initializer='he_uniform', activation=tf.nn.relu , name='Conv1D_3')(maxPool1)
    maxPool2 = MaxPooling1D(pool_size=hyper_params['nni_network/MaxPooling1D_kernel_size_3/randint'], name='MaxPooling1D_3')(conv2)

    conv3 = Conv1D(filters=hyper_params['nni_network/Conv1D_filters_4/randint'], kernel_size=hyper_params['nni_network/Conv1D_kernel_size_4/randint'], kernel_initializer='he_uniform', activation=tf.nn.relu , name='Conv1D_4')(maxPool2)
    
    flatten = Flatten()(conv3)
    dp0 = Dropout(rate=hyper_params['nni_network/Dropout_rate_1/uniform'])(flatten)

    d0 = Dense(hyper_params['nni_network/CNN_Dense_1/randint'], activation=tf.nn.relu, name='Dense_1')(dp0)
    dp1 = Dropout(rate=hyper_params['nni_network/Dropout_rate_2/uniform'])(d0)
    
    d1 = Dense(n_classes,  name='Dense_2')(dp1)

    model = tf.keras.Model(inputs=inp, outputs=d1)

    optimizer = Adam(lr=hyper_params['nni_network/lr/quniform'])

    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=optimizer,
              metrics=['accuracy'])

    return model

def create_MLinApp_model(args, hyper_params, timesteps, input_dim, n_classes):
    inp = tf.keras.Input(shape=(input_dim, timesteps))
    conv0 = Conv1D(filters=hyper_params['nni_network/Conv1D_filters_1/randint'], kernel_size=hyper_params['nni_network/Conv1D_kernel_size_1/randint'], activation=tf.nn.relu, kernel_initializer='he_uniform', input_shape=(input_dim, timesteps), name='Conv1D_1')(inp)
    maxPool0 = MaxPooling1D(pool_size=hyper_params['nni_network/MaxPooling1D_kernel_size_1/randint'], name='MaxPooling1D_1')(conv0)

    conv1 = Conv1D(filters=hyper_params['nni_network/Conv1D_filters_2/randint'], kernel_size=hyper_params['nni_network/Conv1D_kernel_size_2/randint'], kernel_initializer='he_uniform', activation=tf.nn.relu, name='Conv1D_2')(maxPool0)
    maxPool1 = MaxPooling1D(pool_size=hyper_params['nni_network/MaxPooling1D_kernel_size_2/randint'], name='MaxPooling1D_2')(conv1)
 
    conv2 = Conv1D(filters=hyper_params['nni_network/Conv1D_filters_3/randint'], kernel_size=hyper_params['nni_network/Conv1D_kernel_size_3/randint'], kernel_initializer='he_uniform', activation=tf.nn.relu , name='Conv1D_3')(maxPool1)
    maxPool2 = MaxPooling1D(pool_size=hyper_params['nni_network/MaxPooling1D_kernel_size_3/randint'], name='MaxPooling1D_3')(conv2)

    conv3 = Conv1D(filters=hyper_params['nni_network/Conv1D_filters_4/randint'], kernel_size=hyper_params['nni_network/Conv1D_kernel_size_4/randint'], kernel_initializer='he_uniform', activation=tf.nn.relu , name='Conv1D_4')(maxPool2)
    
    flatten = Flatten()(conv3)

    d0 = Dense(hyper_params['nni_network/CNN_Dense_1/randint'], activation=tf.nn.relu, name='Dense_1')(flatten)
    d1 = Dense(n_classes, name='Dense_2')(d0)

    model = tf.keras.Model(inputs=inp, outputs=d1)

    optimizer = Adam(lr=hyper_params['nni_network/lr/quniform'])

    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=optimizer,
              metrics=['accuracy'])
    model.summary() 
    return model

def run(args, params):
    
    miRna_data, miRna_label = load_dataset(data_file, label_file)
    assert np.isnan(miRna_data).sum() == 0
    miRna_data, miRna_label = duplicate_single_values(miRna_data, miRna_label)

    miRna_label = label_processing2(miRna_label)

    # SplittinSg dataset in train, validation, test into %60, %20, %20
    train_data, test_data, train_label, test_label = train_test_split(miRna_data, miRna_label, test_size=0.20, stratify=miRna_label,
                                                                      random_state=42)

    train_data, validation_data, train_label, validation_label = train_test_split(train_data, train_label, test_size=0.25, stratify=train_label,
                                                                      random_state=42)

    timesteps = 1
    input_dim = train_data.shape[1]
    n_classes = len(np.unique(train_label))

    print(f"Shape of train data={train_data.shape} with {len(np.unique(train_label))} classes")
    print(f"Shape of val data= {validation_data.shape} with {len(np.unique(validation_label))} classes")
    print(f"Shape of test data ={test_data.shape} with {len(np.unique(test_label))} classes")
    print(f"Dropout:{dropout_flag}")
  


    settings = { 'activation': 'relu'}
    if dropout_flag == True:
      model = create_MLinApp_model_dropout(args, params, timesteps, input_dim, n_classes)
    else:
      model = create_MLinApp_model(args, params, timesteps, input_dim, n_classes)

    batch_size = params['nni_network/batch_size/randint']

    # training
    history = model.fit(train_data,
                        train_label,
                        validation_data=(validation_data, validation_label),
                        batch_size=batch_size,
                        epochs=args.epochs,
                        verbose=0,
                        callbacks=[SendMetrics(), TensorBoard(log_dir=TENSORBOARD_DIR), ModelCheckpoint(filepath=out_dir+nni.get_experiment_id()+'_'+nni.get_trial_id()+'_'+'best_train/'+'best_train_'+nni.get_experiment_id()+'_'+nni.get_trial_id(), monitor="val_accuracy", save_best_only=True, save_weights_only=True)])

    # test
    model.load_weights(out_dir+nni.get_experiment_id()+'_'+nni.get_trial_id()+'_'+'best_train/'+'best_train_'+nni.get_experiment_id()+'_'+nni.get_trial_id())
    _, acc = model.evaluate(test_data, test_label, verbose=0)
    
    report_file = report_result(args,acc*100, 'test')
    with open(report_file, 'r') as f:
        if acc*100 >= np.max(np.asarray([(line.strip()) for line in f], dtype=np.float64)):
            model.save_weights(out_dir+'best_test/'+'best_test_'+nni.get_experiment_id())
    model.save(out_dir+"model_cnn")
    LOG.debug('Final result is: %d', acc)
    LOG.debug(print(f"Final validation accuracy: {100 * history.history['val_accuracy'][-1]:.2f}%"))
    LOG.debug(print(f"Best validation accuracy: {100 * np.max(history.history['val_accuracy']):.2f}%"))
    LOG.debug(print(f"Test accuracy from training with best validation accuracy: {100 * acc:.2f}%"))
    model_json = model.to_json()
    with open(out_dir+"model.json", "w") as json_file:
        json_file.write(model_json)
    with open(out_dir+"dataset_info.txt", "w") as info:
        info.write(f"Dataset {dataset_name} from: {folder_name}\n")
    
    nni.report_final_result(acc*100)

    shutil.rmtree(out_dir+nni.get_experiment_id()+'_'+nni.get_trial_id()+'_'+'best_train/')


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--network_type", type=str, default=network_type, help="network type", required=False)
    PARSER.add_argument("--datafile", type=str, default=data_file, help="data file", required=False)
    PARSER.add_argument("--labelfile", type=str, default=label_file, help="label file", required=False)
    PARSER.add_argument("--epochs", type=int, default=eps, help="Train epochs", required=False)
    PARSER.add_argument("--filename", type=str, default=searchspace_path, help="File name for search space", required=False)
    PARSER.add_argument("--id", type=str, default=nni.get_experiment_id(), help="Experiment ID", required=False)

    ARGS, UNKNOWN = PARSER.parse_known_args()



    LOG = logging.getLogger('MLinApp_'+ARGS.network_type+'_'+dataset_name)
    LOG.setLevel(logging.DEBUG)

    out_dir = '../output/tmp_' + network_type + '_' + nni.get_experiment_id() + '_' + dataset_name + '/'
    os.environ['NNI_OUTPUT_DIR'] = out_dir
    TENSORBOARD_DIR = os.environ['NNI_OUTPUT_DIR']

    try:

        n_tr = 100
        if (nni.get_sequence_id() > 0) & (nni.get_sequence_id()%n_tr == 0):
            updater.update_searchspace(ARGS) # it will use ARGS.filename to update the search space

        PARAMS = nni.get_next_parameter()       
        run(ARGS, PARAMS)
    
    except Exception as e:
        LOG.exception(e)
        raise