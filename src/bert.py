from transformers import BertTokenizer, TFBertModel
import numpy.core.multiarray as multiarray
import json
import itertools
import multiprocessing
import pickle
from sklearn import svm
from sklearn import metrics as sk_metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import keras
from tensorflow.python.ops import math_ops
from keras import *
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.utils import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.model_selection import StratifiedKFold as SKF

os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Function to save best result
def save_func(file_path,values):
    file=[i.rstrip().split(',') for i in open(file_path).readlines()]
    file.append(values)
    file=pd.DataFrame(file)
    file.to_csv(file_path,header=None,index=None)
        
# Sigmoid Results to Binary    
def sigmoid_to_binary(predicted_labels):
    binary_labels=[]
    for i in predicted_labels:
        if i>0.5:
            binary_labels.append(1)
        else:
            binary_labels.append(0)
    return binary_labels

# Create One Hot Layer
def OneHot(input_dim=None, input_length=None):
    def one_hot(x, num_classes):
        return K.one_hot(K.cast(x, 'uint8'),num_classes=num_classes)

    ## Lambda Layer allows to create a special layer that represents the result of some kind of operation over the data
    return Lambda(one_hot, arguments={'num_classes': input_dim}, input_shape=(input_length,))

# Data Conversion using Dictionary
def data_conversion(dataset,dictionary,max_length):
    matrix=np.zeros([len(dataset),max_length])
    for i in range(len(dataset)):
        dataset[i][1]=list(dataset[i][1])
        for j in range(len(dataset[i][1])):
            for k in dictionary:
                if dataset[i][1][j]==k:
                    dataset[i][1][j]=dictionary.get(k)
        if len(dataset[i][1])<max_length:
            matrix[i,0:len(dataset[i][1])]=dataset[i][1]
        else:
            matrix[i,0:max_length]=dataset[i][1][0:max_length]
    return matrix.astype('int32')

# Generate Covolutional Layers
def generate_cov1D(num_filters,filter_window,stride,padding_method,act_func):
    cov_layer=Conv1D(filters=num_filters,kernel_size=filter_window,strides=stride,padding=padding_method,activation=act_func)
    return cov_layer

# Generate Fully Connect Layers
def generate_fc(num_neurons,act_func):
    fc_layer=Dense(units=num_neurons,activation=act_func)
    return fc_layer

# Generate embedding layers (needs to be the first layer)
def generate_embedding(input_dim,output_dim,input_len):
    embedding=Embedding(input_dim=input_dim+1,output_dim=output_dim,input_length=input_len)
    return embedding

from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import confusion_matrix

def grid_search(prot_train, smile_train, labels_train, prot_test, smile_test, labels_test,
                number_cov_layers, number_fc_layers, prot_seq_len, smile_len,
                prot_dict_size, smile_dict_size, encoding_type, embedding_size,
                num_filters, drop_rate, batch, learning_rate,
                prot_filter_1_window, prot_filter_2_window, prot_filter_3_window,
                prot_filter_4_window, prot_filter_5_window,
                smile_filter_1_window, smile_filter_2_window, smile_filter_3_window,
                smile_filter_4_window, smile_filter_5_window,
                fc_1_size, fc_2_size, fc_3_size, fc_4_size,
                act_func_conv, fc_act_func, epochs, loss_func, output_act, metric_type,
                tokenizer, max_length, num_classes):

    for n_filter in num_filters:
        for d_rate in drop_rate:
            for l_rate in learning_rate:
                for smile_filter_1 in smile_filter_1_window:
                    for smile_filter_2 in smile_filter_2_window:
                        for smile_filter_3 in smile_filter_3_window:
                            for prot_filter_1 in prot_filter_1_window:
                                for prot_filter_2 in prot_filter_2_window:
                                    for prot_filter_3 in prot_filter_3_window:
                                        for fc_neurons_1 in fc_1_size:
                                            for fc_neurons_2 in fc_2_size:
                                                for fc_neurons_3 in fc_3_size:
                                                    # Create a unique filename for each parameter combination
                                                    file_name = f"{n_filter}_{d_rate}_{l_rate}_{smile_filter_1}_{smile_filter_2}_{smile_filter_3}_{prot_filter_1}_{prot_filter_2}_{prot_filter_3}_{fc_neurons_1}_{fc_neurons_2}_{fc_neurons_3}"

                                                    print(f"Training model with configuration: {file_name}")

                                                    # Call bert_classifier with the necessary parameters
                                                    model = bert_classifier(tokenizer, max_length, num_classes, dropout_rate=d_rate)

                                                    # Train the model
                                                    model.fit([prot_train, smile_train], labels_train, batch_size=batch, epochs=epochs, verbose=1)

                                                    # Evaluate the model on the test set
                                                    predicted_labels = model.predict([prot_test, smile_test])
                                                    binary_labels = sigmoid_to_binary(predicted_labels)
                                                    cm = confusion_matrix(labels_test, np.array(binary_labels))

                                                    # Calculate metrics
                                                    metric_values = metrics_function(True, True, True, True, False, False, binary_labels, predicted_labels, labels_test, cm)

                                                    # Save results to CSV
                                                    save_func('../Results_CNN_FCNN.csv', [
                                                        n_filter, d_rate, batch, l_rate, smile_filter_1, smile_filter_2, smile_filter_3,
                                                        prot_filter_1, prot_filter_2, prot_filter_3, fc_neurons_1, fc_neurons_2,
                                                        fc_neurons_3, epochs, loss_func, output_act, act_func_conv, fc_act_func,
                                                        metric_values[0].strip('Sensitivity:'),
                                                        metric_values[1].strip('Specificity:'),
                                                        metric_values[2].strip('F1_Score:'),
                                                        metric_values[3].strip('Accuracy:')
                                                    ])
def generate_input(shape_size,dtype):
    data_input=Input(shape=(shape_size,),dtype=dtype)
    return data_input

# Generate Max or Average Pooling Layers
def generate_pooling(type,pool_size):
    if type=='max':
        max_pool=MaxPooling1D(pool_size=pool_size,padding='valid')
        return max_pool
    elif type=='average':
        average_pool=AveragePooling1D(pool_size=pool_size,padding='valid')
        return average_pool

# List of optimizers with different learning rates
def generate_optimizers(lr_rate):
    optimizer=[Adam(lr=lr_rate),SGD(lr=lr_rate),RMSprop(lr=lr_rate),Adamax(lr=lr_rate),Nadam(lr=lr_rate)]
    return optimizer

#Classifier Metrics
#Sensitivity
def sensitivity(y_true,y_pred):
    y_pred=math_ops.round(y_pred)
    TP = tf.math.count_nonzero(y_pred * y_true)
    TN = tf.math.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.math.count_nonzero(y_pred * (y_true - 1))
    FN = tf.math.count_nonzero((y_pred - 1) * y_true)
    metric=tf.divide(TP,TP+FN)
    return metric
# Specificity
def specificity(y_true,y_pred):
    y_pred=math_ops.round(y_pred)
    TP = tf.math.count_nonzero(y_pred * y_true)
    TN = tf.math.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.math.count_nonzero(y_pred * (y_true - 1))
    FN = tf.math.count_nonzero((y_pred - 1) * y_true)
    metric=tf.divide(TN,TN+FP)
    return metric
# F1-Score
def f1_score(y_true,y_pred):
    y_pred=math_ops.round(y_pred)
    TP = tf.math.count_nonzero(y_pred * y_true)
    TN = tf.math.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.math.count_nonzero(y_pred * (y_true - 1))
    FN = tf.math.count_nonzero((y_pred - 1) * y_true)
    metric=tf.divide(TN,TN+FP)
    precision = tf.divide(TP,TP + FP)
    sensitivity = tf.divide(TP,TP+FN)
    metric = tf.divide(tf.multiply(2*precision,sensitivity),precision + sensitivity)
    return metric

# Metrics Function: Sensitivity, Specificity, F1-Score, Accuracy and AUC
def metrics_function(sensitivity,specificity,f1,accuracy,auc_value,auprc_value,binary_labels,predicted_labels,labels_test,confusion_matrix):
    sensitivity_value=confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0])
    specificity_value= confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
    precision=confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
    f1_value=2*(precision*sensitivity_value)/(precision+sensitivity_value)
    accuracy=accuracy_score(labels_test,np.array(binary_labels))
    auc=roc_auc_score(labels_test,predicted_labels)
    auprc=average_precision_score(labels_test,predicted_labels)
    metrics=[]
    if sensitivity:
        metrics.append('Sensitivity:'+str(sensitivity_value))
    if specificity:
        metrics.append('Specificity:'+str(specificity_value))
    if f1:
        metrics.append('F1_Score:'+str(f1_value))
    if accuracy:
        metrics.append('Accuracy:'+str(accuracy))
    if auc_value:
        metrics.append('AUC:'+str(auc))
    if auprc_value:
        metrics.append('AUPRC: '+str(auprc))
    return metrics   
# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize sequences
def tokenize_data(sequences, max_length):
    tokenized_data = tokenizer(
        sequences,
        padding='max_length',  # Pad sequences to the same length
        max_length=max_length,
        truncation=True,       # Truncate longer sequences
        return_tensors='tf'    # Return as TensorFlow tensors
    )
    return tokenized_data
# Load pre-trained BERT model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Convert input data to BERT embeddings
def get_bert_embeddings(input_ids, attention_mask):
    embeddings = bert_model(input_ids, attention_mask=attention_mask).last_hidden_state
    return embeddings
import tensorflow as tf
from transformers import TFBertModel
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model

def bert_classifier(tokenizer, max_length, num_classes, dropout_rate=0.2):

    # Define the BERT model
    input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")

    # Use pre-trained BERT model
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    bert_output = bert_model(input_ids, attention_mask=attention_mask)
    
    # Get embeddings from BERT's output
    cls_token = bert_output.last_hidden_state[:, 0, :]  # CLS token representation
    
    # Dropout for regularization
    dropout = Dropout(dropout_rate)(cls_token)
    
    # Fully connected layers for classification
    dense = Dense(128, activation='relu')(dropout)
    dense = Dropout(dropout_rate)(dense)
    
    # Output layer for classification
    output = Dense(num_classes, activation='softmax')(dense)

    # Define the model
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model


# Other metrics (sensitivity, specificity, etc.)
# You can keep the same metrics you used before with minor adjustments to use the predicted labels
import json
import numpy as np
import tensorflow as tf

def main():
    ## Load Protein Sequences
    prot_train = [i.rstrip().split(',') for i in open('../Datasets/Protein_Train_Dataset.csv')]
    prot_test = [i.rstrip().split(',') for i in open('../Datasets/Protein_Test_Dataset.csv')]

    ## Load SMILE Strings
    drug_train = [i.rstrip().split(',') for i in open('../Datasets/Smile_Train_Dataset.csv')]
    drug_test = [i.rstrip().split(',') for i in open('../Datasets/Smile_Test_Dataset.csv')]

    ## Load Protein & Smile Dictionaries
    prot_dictionary = json.load(open('../Dictionaries/aa_properties_dictionary.txt'))
    smile_dictionary = json.load(open('../Dictionaries/smile_dictionary.txt'))

    ## Convert sequences and SMILES to integers
    prot_train_data = data_conversion(prot_train, prot_dictionary, 300)
    prot_test_data = data_conversion(prot_test, prot_dictionary, 300)

    smile_train_data = data_conversion(drug_train, smile_dictionary, 40)
    smile_test_data = data_conversion(drug_test, smile_dictionary, 40)

    ## Load labels
    labels_train = np.load('../Labels/labels_train.npy')
    labels_test = np.load('../Labels/labels_test.npy')

    # Set hyperparameters
    prot_seq_len = 300    # Sequence length for proteins
    smile_len = 40        # Sequence length for SMILES
    prot_dict_size = len(prot_dictionary)
    smile_dict_size = len(smile_dictionary)
    num_filters = [16, 32, 48]  # Number of convolutional filters
    embedding_size = [0]  # Size of embedding vector (0 if not used)
    prot_filter_window = [2, 3]  # Filter sizes for protein convolution
    smile_filter_window = [2, 3]  # Filter sizes for SMILE convolution
    act_func_conv = 'relu'  # Activation function for convolutional layers
    fc_act_func = 'relu'  # Activation function for fully connected layers
    drop_rate = [0.3, 0.5]  # Dropout rates
    batch = 128  # Batch size
    epochs = 100  # Number of epochs
    fc_size = [32, 64, 128]  # Sizes for fully connected layers
    number_cov_layers = 3  # Number of convolution layers
    number_fc_layers = 3  # Number of fully connected layers
    learning_rate = [0.001, 0.01]  # Learning rates
    loss_func = 'binary_crossentropy'  # Loss function
    output_act = 'sigmoid'  # Output activation function
    metric_type = ['accuracy', sensitivity, specificity, f1_score]  # Metrics
    encoding_type = 'one_hot'  # Encoding type for input
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 128  # Define max sequence length
    num_classes = 2 
    # Perform grid search with reduced parameters
    grid_search(
        prot_train_data, smile_train_data, labels_train,
        prot_test_data, smile_test_data, labels_test,
        number_cov_layers, number_fc_layers, prot_seq_len, smile_len, 
        prot_dict_size, smile_dict_size, encoding_type, embedding_size,
        num_filters, drop_rate, batch, learning_rate, 
        prot_filter_window, prot_filter_window, prot_filter_window, 
        [0], [0], smile_filter_window, smile_filter_window, smile_filter_window,
        [0], [0], fc_size, fc_size, fc_size, [0],
        act_func_conv, fc_act_func, epochs, loss_func, output_act, metric_type,
        tokenizer, max_length, num_classes
    )
    


    # Save best model
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', 
                                                          monitor='val_loss', 
                                                          save_best_only=True, 
                                                          save_freq='epoch')

if __name__ == '__main__':
    main()
