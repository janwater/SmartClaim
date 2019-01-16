
"""
Created on Fri Dec 15 09:34:56 2017

@author: E154709

# ==============================================================================

A deep KerasClassifier wrapper

https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-
# import standard python modules
import argparse
import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
#from sklearn.utils.validation import column_or_1d
from sklearn.pipeline import Pipeline
import timeit

# deep learning modules
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D


# specific module from file
from csModel import _plot_ROC_curve
from csModel import _class_info
from csModel import _verify_claim
from csModel import _report_class

from keras.utils.np_utils import to_categorical


seed = 7
np.random.seed(seed)
FLAGS = None

# user defined functions
def _create_filename(path, fname, affix = '.csv'):
    # create new file name
    output_name = fname + affix        
    foutput_name = os.path.join(path, '', output_name)
             
    return foutput_name

def _read_inputTags():
    # read input tags
    inputTags = ['numClaims',
               'numSales',
               'dollarPerClaim',
               'dollarPerSale',
               'claimRatio',
               'INVENTORY_STYLE_CD',
               'INVENTORY_SIZE_CD',
               'INVENTORY_BACKING_CD',
               'INVENTORY_COLOR_CD'
               ]
    return inputTags

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    data = np.asarray(data)
    labels = np.asarray(labels)
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def normalize_data(x, scaler = None):
    # normalize data
    xvalue = x.values 
    if scaler is None:
        scaler = preprocessing.StandardScaler().fit(xvalue) 
        
    #normX_scaler = preprocessing.MinMaxScaler().fit(xvalue) 
    xScaled = scaler.transform(xvalue)   
    xn = pd.DataFrame(xScaled)
    xn.index = x.index
    xn.columns = x.columns
    
    return xn, scaler

# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(15, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model_json = model.to_json()
    with open("SmartClaim_dlnn.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights('SmartClaim_dlnn.h5')
    
    return model

# larger model
def create_larger():
    # create model
    model = Sequential()
    model.add(Dense(15, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   
    
    model_json = model.to_json()
    with open("SmartClaim_dlnn2.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('SmartClaim_dlnn2.h5')
    
    return model

# MLP  model
def create_MLP():
    # create MLP model
    model = Sequential()
    model.add(Dense(64, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])   
    
    model_json = model.to_json()
    with open("SmartClaim_mlp.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('SmartClaim_mlp.h5')
    
    return model

# CNN  model
def create_CNN():
    # create MLP model
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(None,9)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', 
                  optimizer='rmsprop', 
                  metrics=['accuracy'])   
    
    model_json = model.to_json()
    with open("SmartClaim_cnn.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('SmartClaim_cnn.h5')
    
    return model

from keras.callbacks import TensorBoard
from tensorboard.plugins.pr_curve import summary as pr_summary

# Check complete example in:
# https://github.com/akionakamura/pr-tensorboard-keras-example
class PRTensorBoard(TensorBoard):
    def __init__(self, *args, **kwargs):
        # One extra argument to indicate whether or not to use the PR curve summary.
        self.pr_curve = kwargs.pop('pr_curve', True)
        super(PRTensorBoard, self).__init__(*args, **kwargs)

        global tf
        import tensorflow as tf

    def set_model(self, model):
        super(PRTensorBoard, self).set_model(model)

        if self.pr_curve:
            # Get the prediction and label tensor placeholders.
            predictions = self.model._feed_outputs[0]
            labels = tf.cast(self.model._feed_targets[0], tf.bool)
            # Create the PR summary OP.
            self.pr_summary = pr_summary.op(tag='pr_curve',
                                            predictions=predictions,
                                            labels=labels,
                                            display_name='Precision-Recall Curve')

    def on_epoch_end(self, epoch, logs=None):
        super(PRTensorBoard, self).on_epoch_end(epoch, logs)

        if self.pr_curve and self.validation_data:
            # Get the tensors again.
            tensors = self.model._feed_targets + self.model._feed_outputs
            # Predict the output.
            predictions = self.model.predict(self.validation_data[:-2])
            # Build the dictionary mapping the tensor to the data.
            val_data = [self.validation_data[-2], predictions]
            feed_dict = dict(zip(tensors, val_data))
            # Run and add summary.
            result = self.sess.run([self.pr_summary], feed_dict=feed_dict)
            self.writer.add_summary(result[0], epoch)
        self.writer.flush()
        

#defube a function to output validation and testing results
def _output_val_test(clf, name, nxValid, yValid, xnTest, yTest):
    
    # validation data
    print("\n Validation...")
    pred = _report_class(clf, name, nxValid, yValid)
    pred = pd.DataFrame(data=pred, index=yValid.index, columns =['Pred'])
    print(pred.head(10), pred.shape)
    pred['Index'] = 'val'
    val = pd.concat([nxValid, pred], axis = 1)
    
    # Testing data
    print("\n Testing...")    
    test = _report_class(clf, name, xnTest, yTest)
    test = pd.DataFrame(data=test, index= yTest.index, columns =['Pred'])
    print(test.head(10), test.shape)
    test['Index'] = 'tst'
    tst = pd.concat([xnTest, test], axis = 1)
    
    output = pd.concat([val, tst], axis = 0)
    print(output.shape, output.head(5))
    
    return output

# main program to load data, build model, validation, testing...
def main(_):
  # Import data
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # set working directory
    parentDir = 'C:/Users/E154709/Desktop/Mohawk/Claims'
    os.chdir(parentDir)
    #modeling data
    dataPath = parentDir + '/Data/'
    if not os.path.exists(dataPath):
        os.mkdir(dataPath)
    
    #create data file
    fname = 'modelData2'
    fmodelData = _create_filename(dataPath,fname)
    modelData = pd.read_csv(fmodelData, encoding='latin1') # Read the data
    
    # import input tags
    inputTags = _read_inputTags()
    yTags = 'Claims'
    
    # read model data
    Xin = modelData[inputTags]
    Yout = modelData[yTags]
    
    # split data    
    ts = 0.3
    xTrain, xTest, yTrain, yTest = train_test_split(Xin, Yout, stratify=Yout, test_size=ts, random_state=531) 
    xnTrain, scaler = normalize_data(xTrain)
    xnTest, scaler = normalize_data(xTest, scaler)
    
    #split the training data into modeling set and validation set
    nxModel, nxValid, yModel, yValid = train_test_split(xnTrain, yTrain, stratify=yTrain, test_size=ts, random_state=531)
    xModel, xValid, yModel, yValid = train_test_split(xTrain, yTrain, stratify=yTrain, test_size=ts, random_state=531)

    # <codecell>
    # class information
    _class_info(yTrain)
    _class_info(yTest)
     
    #yMod_binary = to_categorical(yModel)
    #yVal_binary = to_categorical(yValid)
    #yTst_binary = to_categorical(yTest)
    #xModel_trn = nxModel.reset_index(drop=True)
    #yrTrain = yModel.reset_index(drop=True)
    #modeling

    # <codecell>
    print("\n Preliminary Deep Learning model ...")
    start_time1 = timeit.default_timer() 
    #evaluate model with standardized dataset
    pre_dlnn_clf = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=150, verbose=0)
        
    #clf_pre_dlnn = _plot_ROC_curve(pre_dlnn_clf, nxModel, yModel)
    name = 'PRE_DLNN'
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    pre_results = cross_val_score(pre_dlnn_clf, nxModel.values, yModel.values, cv=kfold)
    print("Results: %.2f%% (%.2f%%)" % (pre_results.mean()*100, pre_results.std()*100))
    print(pre_results)
    pre_dlnn_clf.fit(nxModel.values, yModel.values)
    dlnn_output = _output_val_test(pre_dlnn_clf, name, nxValid, yValid, xnTest, yTest)
    print(dlnn_output.head(2))
    
    elapsed1 = timeit.default_timer() - start_time1
    print("CPU Execution time ... %d" %elapsed1)
    #dlnn_output.to_csv('dlnn_output.csv')
    
    # evaluate baseline model with standardized dataset
# <codecell>
    
    print("\n Larger Deep Learning model ...")
    start_time2 = timeit.default_timer()
    larger_dlnn_clf = KerasClassifier(build_fn=create_larger, nb_epoch=100, batch_size=150, verbose=0)
        
    #clf_pre_dlnn = _plot_ROC_curve(pre_dlnn_clf, nxModel, yModel)
    name = 'LARGER_DLNN'
    
    """
    np.random.seed(seed)
    estimators = []
    estimators.append(('standardize', preprocessing.StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(pipeline, xModel.values, yModel.values, cv=kfold)
      
    """
    larger_results = cross_val_score(larger_dlnn_clf, nxModel.values, yModel.values, cv=kfold)
    
    print("Larger: %.2f%% (%.2f%%)" % (larger_results.mean()*100, larger_results.std()*100))
    larger_dlnn_clf.fit(nxModel.values, yModel.values)
    dlnn_output2 = _output_val_test(larger_dlnn_clf, name, nxValid, yValid, xnTest, yTest)
    print(dlnn_output2.head(2))
    elapsed2 = timeit.default_timer() - start_time2
    print("CPU Execution time ... %d" %elapsed2)
    
# <codecell>
    
    print("\n MLP Deep Learning model ...")
    start_time3 = timeit.default_timer()
    mlp_dlnn_clf = KerasClassifier(build_fn=create_MLP, nb_epoch=100, batch_size=150, verbose=0)
        
    #clf_pre_dlnn = _plot_ROC_curve(pre_dlnn_clf, nxModel, yModel)
    name = 'MLP_DLNN'
    
    mlp_results = cross_val_score(mlp_dlnn_clf, nxModel.values, yModel.values, cv=kfold)
    
    print("MLP_DLNNr: %.2f%% (%.2f%%)" % (mlp_results.mean()*100, mlp_results.std()*100))
    mlp_dlnn_clf.fit(nxModel.values, yModel.values)
    dlnn_output3 = _output_val_test(mlp_dlnn_clf, name, nxValid, yValid, xnTest, yTest)
    print(dlnn_output3.head(2))
    elapsed3 = timeit.default_timer() - start_time3
    print("CPU Execution time ... %d" %elapsed3)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='C:/Users/E154709/Desktop/Mohawk/Claims',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)