import numpy as np
import csv
import json



import numpy as np
import pickle as pk  

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn import svm
import sklearn.metrics
import csv, pprint
import scipy.stats
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import math
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE

random_state = 42
test_size = 0.2

def load_dataset(data_file_name, label_file_name):
    miRna_label = extract_label(label_file_name)
    miRna_data = np.genfromtxt(data_file_name, delimiter=',')
    print(f"Dataset dimensions: {miRna_data.shape[0]}")
    return miRna_data, miRna_label

def load_dataset_npy(data_file_name, label_file_name):

    # miRna_label = extract_label(label_file_name)
    miRna_label = np.load(label_file_name, allow_pickle=True)
    # miRna_data = np.genfromtxt(data_file_name, delimiter=',', dtype="float")
    miRna_data = np.load(data_file_name, allow_pickle=True)
    print(f"Dataset dimensions: {miRna_data.shape[0]}")
    return miRna_data, miRna_label

# def extract_label(file_name):
#     label = []
#     with open(file_name, "r") as fin:
#         reader = csv.reader(fin, delimiter=',')
#         first = True
#         for row in reader:
#             label.append(row)
#     return np.array(label)

def extract_label(file_name, verbose=False):
    data = {}
    label = []
    with open(file_name, "r") as fin:
        reader = csv.reader(fin, delimiter=',')
        first = True
        for row in reader:
            lbl = row[2]
            if first or "TARGET" in lbl:
                first = False
                continue
            lbl = lbl.replace("TCGA-","")

            label.append(lbl)
            if lbl in data.keys():
                data[lbl] += 1 
            else:
                data[lbl] = 1
    if verbose:
        print(f"Number of classes in the dataset = {len(data)}")
        pprint.pprint(data, indent=4)

    return label

def create_dictionary(labels):
    dictionary = {}
    class_names = np.unique(labels)
    for i, name in enumerate(class_names):
        dictionary[name] = i
    return dictionary

def label_to_binary_matrix(labels, dictionary=None):
    getPreKnowledge = False
    number_of_class = np.unique(labels).size
    new_miRna_label = np.zeros((labels.shape[0], number_of_class))
    
    if dictionary == None: 
        dictionary = create_dictionary(labels)    
    else:
        getPreKnowledge=True

    for i in range(new_miRna_label.shape[0]):
        current_label = labels[i]
        new_miRna_label[i, dictionary[current_label[0]]] = 1

    if getPreKnowledge == False:
        return new_miRna_label, dictionary

    return new_miRna_label


# data_file = '../data/superclass3/data.csv'
# label_file = '../data/superclass3/labels.csv'


## SET NETWORK TYPE:
# network_type = 'cnn'
super_class = 3
# data_file_extension = ".npy"
# data_folder = f'../../data/superclass{super_class}/'


miRna_label = extract_label("E:\Il mio Drive\MLinApp_project_mine\mnist_onehot/txt/tcga_mir_label.csv")
miRna_data = np.genfromtxt('E:\Il mio Drive\MLinApp_project_mine\mnist_onehot/txt/tcga_mir_rpm.csv', delimiter=',')[1:,0:-1]

# Deleting all the data that came from TARGET-* labels
number_to_delete = abs(len(miRna_label) - miRna_data.shape[0])
miRna_data = miRna_data[number_to_delete:, :]

# Z-score normalization
miRna_data = scipy.stats.zscore(miRna_data, axis=1) # np.array -> shape = (n_samples, n_features)
miRna_label = np.array(miRna_label, dtype="U24") # np.array -> shape = (n_samples, )

assert np.isnan(miRna_data).sum() == 0

if super_class == 3:
  miRna_label = np.ravel(miRna_label)
  ros = RandomOverSampler(random_state=random_state)
#   miRna_data, miRna_label = ros.fit_resample(miRna_data, miRna_label)
  # miRna_data, miRna_label = SMOTE().fit_resample(miRna_data, miRna_label)
  # miRna_data, miRna_label = ADASYN().fit_resample(miRna_data, miRna_label)
  miRna_data, miRna_label = SVMSMOTE().fit_resample(miRna_data, miRna_label)
  miRna_label = miRna_label.reshape(-1,1)

train_data, test_data, train_label, test_label = train_test_split(miRna_data, miRna_label, test_size=test_size, random_state=random_state)

# train_data, train_label = load_dataset_npy(train_data_file, train_label_file)
# test_data, test_label = load_dataset_npy(test_data_file, test_label_file)

# train_label = train_label.reshape((-1, 1))
# test_label = test_label.reshape((-1, 1))

# train_label_shape = train_label.shape
# test_label_shape = test_label.shape

# label_binary_matrix, local_label_mapping = label_to_binary_matrix(np.vstack((train_label, test_label)))

# train_label_binary_matrix = label_binary_matrix[0:train_label_shape[0], :]
# test_label_binary_matrix = label_binary_matrix[train_label_shape[0]:, :]

print()

