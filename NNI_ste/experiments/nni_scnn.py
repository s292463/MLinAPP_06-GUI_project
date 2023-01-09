import csv
import os
import nengo
import nengo_dl
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras import Input
from keras import layers, models
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop
from keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from keras.callbacks import TensorBoard, Callback
import argparse
import logging
import json
import zipfile
import sqlite3
import nni
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, freqz
from nni.tuner import Tuner
from nni.experiment import Experiment
from nni.algorithms.hpo.hyperopt_tuner import HyperoptTuner
from nni.tools.nnictl import updater, nnictl_utils
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL']
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'  

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
rng = np.random.RandomState(seed)
CNN_models_code = {
                  '0': 'Xs68DgU3',
                  '1': 'FnrO3zG8',
                  '2': 'Kw4Amaru',
#                  '2': 'AwqxBIOE', #no do
                  '3': 'zQ402ChL',
                  #'4': 'RgpmIaYS',
                   '4': '0Nm8bK2S',
                  }

### SET NETWORK TYPE:
network_type = 'scnn'


##### DATASET-RELATED SETTINGS #######################


# name to identify experiments on different datasets
superclass = 'superclass4'
dataset_name = 'miRNA-'+superclass
folder_name= '../data-knn'

# file name of the dataset
data_file = '../'+folder_name+'/'+superclass+'/data.csv'
label_file = '../'+folder_name+'/'+superclass+'/labels.csv'

optim_nni_experiment = CNN_models_code[superclass[-1]]

######################################################


searchspace_path = '../searchspaces/nni_SearchSpace_'+network_type+'.json'
eps = 100

def duplicate_single_values(x,y):
  # Duplication in order to have potentially one sample per train, test, validation
  print("Data Analysis for Stratify...")
  count = {}
  data = list(x)
  labels = list(y)
  print("Before duplication:", y.shape[0])
  for label in np.unique(y):
   count[label] = 0
  for label in y:
    count[label] = count[label] + 1
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
  return data,labels



def load_dataset(data_file_name, label_file_name):
    miRna_label = extract_label(label_file_name)
    miRna_data = np.genfromtxt(data_file_name, delimiter=',')
    return duplicate_single_values(miRna_data, miRna_label)


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
    dictionary = create_dictionary(labels)
    for i in range(labels.shape[0]):
      new_miRna_label.append(dictionary[labels[i]])
    return  np.array(new_miRna_label)

def report_result(result, result_type, network_type):

    if result_type == 'test':
        report_file = out_dir + 'nni_' + network_type + '_' + nni.get_experiment_id() + '_' + result_type + '_accs'
    else:
        report_file = out_dir + 'nni_' + network_type + '_' + nni.get_experiment_id() + '_' + result_type + '_accs_' + nni.get_trial_id()
    
    with open(report_file, 'a') as f:
        f.write(str(result))
        f.write('\n')
    
    return report_file


import zipfile,pandas as pd, sqlite3

def get_batch(optim_nni_experiment, dataset_name):
    ##### GET NETWORK STRUCTURE PARAMETERS from previous NNI optimization of non-spiking CNN #####
  optim_db_filepath = "root/nni-experiments/{}/db/nni.sqlite".format(optim_nni_experiment)
  local_experiment_folder = "./results/Experiment_{}_{}/".format("cnn",optim_nni_experiment)
  for ii in os.listdir(local_experiment_folder):
    if optim_nni_experiment in ii:
      target = ii
  zf = zipfile.ZipFile(local_experiment_folder+target)
  con = sqlite3.connect(zf.extract(optim_db_filepath))
  df = pd.read_sql_query("SELECT * from MetricData", con)
  con.close()
  df_default = df[df["type"]=="FINAL"].sort_values(by='data',ascending=False)
  optim_nni_trial = df_default["trialJobId"].iloc[0]
  optim_filename = 'parameter.cfg'
  optim_nni_ref = 'nni-experiments/'+optim_nni_experiment+'/trials/'+optim_nni_trial
  optim_nni_dir = os.path.expanduser('~')
  optim_filepath = os.path.join(optim_nni_dir,optim_nni_ref,optim_filename)

  for ii in os.listdir(local_experiment_folder):
    if optim_nni_experiment in ii:
      target = ii

  zf = zipfile.ZipFile(local_experiment_folder+target)

  with open(zf.extract(optim_filepath[1:]), 'r') as f:
      data = f.read()

  param_data = json.loads(data)
  network_parameters = param_data['parameters']
  return network_parameters['nni_network/batch_size/randint']


def create_nengo_model():

  local_output_folder = "../results/Experiment_{}_{}/".format("cnn",optim_nni_experiment)
  # for ii in os.listdir(local_output_folder):
  #     if "output" in ii:
  #         target = ii
  # zf = zipfile.ZipFile(local_output_folder+target)
  # zf.extractall()

  model = load_model("../output/tmp_{}_{}_{}/model_cnn".format("cnn",optim_nni_experiment,dataset_name))
  print("Model loaded correctly")
  model.summary()

  ### REMEMBER: here the model is only converted into Nengo
  converter = nengo_dl.Converter(model,
                                  max_to_avg_pool=True,
                                )
  return model, converter

class SendMetrics(Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
        #LOG.debug(logs)
        nni.report_intermediate_result(logs['val_probe_accuracy']*100)


def run_nengo(args, params):
    miRna_data, miRna_label = load_dataset(data_file, label_file)

    for label in np.unique(miRna_label):
      print(f"\t{label}, {len(miRna_label[miRna_label==label])}")
    
    miRna_label = label_processing2(miRna_label)

    # Splitting dataset in train, validation, test into %60, %20, %20
    train_data, test_data, train_label, test_label = train_test_split(miRna_data, miRna_label, stratify=miRna_label, test_size=0.20,random_state=42)

    train_data, validation_data, train_label, validation_label = train_test_split(train_data, train_label, test_size=0.25,random_state=42)

    print(f"Number of train classes={len(np.unique(train_label))}")
    print(f"Number of validation classes = {len(np.unique(validation_label))}")
    print(f"Number of test classes = {len(np.unique(test_label))}")
  
    keras_model, converter = create_nengo_model()
    
    train_data = train_data.reshape((train_data.shape[0], 1 , train_data.shape[1]))
    validation_data = validation_data.reshape((validation_data.shape[0], 1 ,validation_data.shape[1]))
    test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))
    
    train_label = train_label.reshape((train_label.shape[0], 1, -1))
    validation_label = validation_label.reshape((validation_label.shape[0], 1, -1))
    test_label = test_label.reshape((test_label.shape[0], 1, -1))
    
    print(test_label.shape)
    # add probes to the convolutional layers, to apply the firing rate regularization
     
    with converter.net:
        nengo_dl.configure_settings(planner=nengo_dl.graph_optimizer.noop_planner)

        output_p = converter.outputs[keras_model.output]
        conv0_p = nengo.Probe(converter.layers[keras_model.layers[1].get_output_at(-1)])
        conv1_p = nengo.Probe(converter.layers[keras_model.layers[3].get_output_at(-1)])
        conv2_p = nengo.Probe(converter.layers[keras_model.layers[5].get_output_at(-1)])
        conv3_p = nengo.Probe(converter.layers[keras_model.layers[7].get_output_at(-1)])

    
    n_steps = params['nni_keras2snn_network/n_steps/randint']

    with nengo_dl.Simulator(converter.net, minibatch_size=params['nni_keras2snn_network/batch_size/randint']) as sim:
        
        nengo_model_summary = sim.keras_model
        nengo_params = sum(np.prod(s.shape) for s in nengo_model_summary.weights)
        nengo_trainable_params = sum(np.prod(w.shape) for w in nengo_model_summary.trainable_weights)
        LOG.debug(print('Total params:','{:,d}'.format(nengo_params)))
        LOG.debug(print('Trainable params:','{:,d}'.format(nengo_trainable_params)))
        LOG.debug(print('Non-trainable params:','{:,d}'.format(nengo_params-nengo_trainable_params)))
        
        # add regularization loss functions to the convolutional layers
        sim.compile(
                    optimizer=tf.optimizers.Adam(params['nni_keras2snn_network/lr/quniform']),
                    loss={
                          output_p:tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                          conv0_p: tf.losses.mse,
                          conv1_p: tf.losses.mse,
                          conv2_p: tf.losses.mse,
                          conv3_p: tf.losses.mse,
                         },
                    loss_weights={
                                  output_p: 1, 
                                  conv0_p: params['nni_keras2snn_network/reg_conv0/quniform'], 
                                  conv1_p: params['nni_keras2snn_network/reg_conv1/quniform'],
                                  conv2_p: params['nni_keras2snn_network/reg_conv2/quniform'],
                                  conv3_p: params['nni_keras2snn_network/reg_conv3/quniform']

                                 },
                    metrics=["accuracy"],
                   )
        
        class CheckPoint(Callback):
            '''
            Keras callback to check training results epoch by epoch
            '''
            def on_epoch_end(self, epoch, logs={}):
                '''
                Run on end of each epoch
                '''
                report_file = report_result(logs['val_probe_accuracy']*100,'validation',args.network_type)

                with open(report_file, 'r') as f:
                    if logs['val_probe_accuracy']*100 >= np.max(np.asarray([(line.strip()) for line in f], dtype=np.float64)):
                        print("Saving train...")
                        sim.save_params(out_dir+'best_train_'+nni.get_experiment_id()+'_'+nni.get_trial_id())

        history = sim.fit(
                          {converter.inputs[keras_model.input]: train_data},
                          {
                           output_p: train_label,
                           conv0_p: np.ones((train_label.shape[0], 1, conv0_p.size_in)) * params['nni_keras2snn_network/target_rate_0/randint'],
                           conv1_p: np.ones((train_label.shape[0], 1, conv1_p.size_in)) * params['nni_keras2snn_network/target_rate_1/randint'],
                           conv2_p: np.ones((train_label.shape[0], 1, conv2_p.size_in)) * params['nni_keras2snn_network/target_rate_2/randint'],
                           conv3_p: np.ones((train_label.shape[0], 1, conv3_p.size_in)) * params['nni_keras2snn_network/target_rate_3/randint'],
                          },
                          validation_data = (validation_data, validation_label),
                          epochs=args.epochs,
                          verbose=0,
                          callbacks=[SendMetrics(), CheckPoint(), TensorBoard(log_dir=TENSORBOARD_DIR)],
                         )

    sim.close()
    
    ### Conversion to spiking 
    
    trained_converter = nengo_dl.Converter(keras_model,
                                           max_to_avg_pool=True,
                                           swap_activations={tf.keras.activations.relu: nengo.SpikingRectifiedLinear()},
                                           scale_firing_rates=params['nni_keras2snn_network/scale_firing_rates/randint'],
                                           synapse=params['nni_keras2snn_network/synapse/quniform'],
                                          )
    
    with trained_converter.net:
        nengo_dl.configure_settings(planner=nengo_dl.graph_optimizer.noop_planner)

        output_p = trained_converter.outputs[keras_model.output]
        conv0_p = nengo.Probe(trained_converter.layers[keras_model.layers[1].get_output_at(-1)])
        conv1_p = nengo.Probe(trained_converter.layers[keras_model.layers[3].get_output_at(-1)])
        conv2_p = nengo.Probe(trained_converter.layers[keras_model.layers[5].get_output_at(-1)])
        conv3_p = nengo.Probe(trained_converter.layers[keras_model.layers[7].get_output_at(-1)])
    
    with nengo_dl.Simulator(trained_converter.net, minibatch_size=params['nni_keras2snn_network/batch_size/randint']) as sim:
        
        sim.compile(
                    optimizer=tf.optimizers.Adam(params['nni_keras2snn_network/lr/quniform']),
                    loss={
                          output_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                          conv0_p: tf.losses.mse,
                          conv1_p: tf.losses.mse,
                          conv2_p: tf.losses.mse,
                          conv3_p: tf.losses.mse,
                         },
                    loss_weights={
                                  output_p: 1, 
                                  conv0_p: params['nni_keras2snn_network/reg_conv0/quniform'], 
                                  conv1_p: params['nni_keras2snn_network/reg_conv1/quniform'],
                                  conv2_p: params['nni_keras2snn_network/reg_conv2/quniform'],
                                  conv3_p: params['nni_keras2snn_network/reg_conv3/quniform']
                                 },
                    metrics=["accuracy"],
                   )
        n_steps = params['nni_keras2snn_network/n_steps/randint']

        tiled_test_data = np.tile(test_data, (1, n_steps, 1))
        
        tiled_test_label = np.tile(test_label, (1, n_steps, 1))
        sim.load_params(out_dir+'best_train_'+nni.get_experiment_id()+'_'+nni.get_trial_id())
        try:
            accuracy = sim.evaluate({trained_converter.inputs[keras_model.input]: tiled_test_data}, {trained_converter.outputs[keras_model.output]:tiled_test_label})['probe_accuracy']
            #data = sim.predict({trained_converter.inputs[keras_model.input]: tiled_test_data})
            #predictions = np.argmax(data[trained_converter.outputs[keras_model.output]][:, -1], axis=-1)
            #accuracy = (predictions[:] == test_label[:predictions.shape[0], 0, 0]).mean()
            #print(predictions[:100])
            report_file = report_result(accuracy*100, 'test', args.network_type)
            with open(report_file, 'r') as f:
                if accuracy*100 >= np.max(np.asarray([(line.strip()) for line in f], dtype=np.float64)):
                    sim.save_params(out_dir+'best_test_'+nni.get_experiment_id())
               
        except Exception as e:
            LOG.exception(e)
            raise

    sim.close()
    
    LOG.debug(print(f"Final validation accuracy: {100 * history.history['val_probe_accuracy'][-1]:.2f}%"))
    LOG.debug(print(f"Best validation accuracy: {100 * np.max(history.history['val_probe_accuracy']):.2f}%"))
    LOG.debug(print(f"Test accuracy from training with best validation accuracy: {100 * accuracy:.2f}%"))
    nni.report_final_result(accuracy*100)

    os.remove(out_dir+'best_train_'+nni.get_experiment_id()+'_'+nni.get_trial_id()+'.npz')
    os.remove(out_dir+'nni_'+args.network_type+'_'+nni.get_experiment_id()+'_validation_accs_'+nni.get_trial_id())

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--network_type", type=str, default=network_type, help="network type", required=False)
    PARSER.add_argument("--datafile", type=str, default=data_file, help="data file", required=False)
    PARSER.add_argument("--epochs", type=int, default=eps, help="Train epochs", required=False)
    PARSER.add_argument("--filename", type=str, default=searchspace_path, help="File name for search space", required=False)
    PARSER.add_argument("--id", type=str, default=nni.get_experiment_id(), help="Experiment ID", required=False)
    
    ARGS, UNKNOWN = PARSER.parse_known_args()

    LOG = logging.getLogger('MLinApp_'+ARGS.network_type+'_'+dataset_name)
    out_dir = '../output/tmp_' + ARGS.network_type + '_' + nni.get_experiment_id() + '_' + dataset_name + '/'
    os.environ['NNI_OUTPUT_DIR'] = out_dir
    TENSORBOARD_DIR = os.environ['NNI_OUTPUT_DIR'] 
    
    try:

        n_tr = 200
        if (nni.get_sequence_id() > 0) & (nni.get_sequence_id()%n_tr == 0):
            updater.update_searchspace(ARGS) # it will use ARGS.filename to update the search space

        PARAMS = nni.get_next_parameter()
        LOG.debug(PARAMS)
        
        run_nengo(ARGS,PARAMS)
    
    except Exception as e:
        LOG.exception(e)
        raise