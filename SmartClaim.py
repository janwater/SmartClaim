# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 08:35:04 2017

@author: E154709
"""

"""
Version 2.0
Feb 20, 2018

Carpet Claim Intelligent Recomendation Modeling
 - Based on cusomter historical data (Sales, Claims, $Sales and $Claims) - R1
 - Product attributes (Style, Size, Backing, & Color)
 - Product historical issues (F1ROLL, F1GROL, F1DLOT)
 - Machine learning
 - Automatic and quantitative recomendation of the claim for approval or declination
 
"""
print(__doc__)

#
# Load necessary libraries
import numpy as np
import pandas as pd
import os, inspect
import glob
import pickle
import sys
import datetime

#%matplotlib inline 
import matplotlib.pylab as plt

#from bokeh.charts import Bar, BoxPlot
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn import mixture

#import xgboost as xgb
#from xgboost.sklearn import XGBClassifier

# import model modules
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# <codecell>
# set working directory
parentDir = 'C:/Users/E154709/Desktop/Mohawk/Claims/SmartClaim'
os.chdir(parentDir)

# load functions from python file
from blagging import BlaggingClassifier

from functions import _pull_hana_hdb
from functions import _create_path
from functions import _create_filename
from functions import _class_info
from functions import _verify_claim
from functions import _plot_ROC_curve
from functions import _replace_inf
from functions import _plot_prob
from functions import _plot_decision_tree
from functions import _plot_confusion_matrix
from functions import _encode_label_
from functions import _convert_float
from functions import _predict_claim
from functions import _output_val_test

# load constants
from constants import INPUT_TAGS

# <codecell>

# extract the basename
fprogram = 'SmartClaim.py'
curFname,file_extension1 = os.path.splitext(fprogram)
print(curFname)

#modeling data
dataPath = parentDir + '/Data/'
_create_path(dataPath)

allFiles = glob.glob(dataPath + "*.csv")
#print(allFiles)
# <codecell>   
# main program to load data, build model, validation, testing...
for finput in allFiles:
    imode = 0
#def main(finput, imode = 1):
        
    """
    # finput: input data file
    # outputPath: output path
    # imode = 0: training, 1: prediction
    """
    
    fbasename = os.path.basename(finput)
    fname, file_extension2 = os.path.splitext(fbasename)
    parentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory    
    print(parentDir)
    
    #output directory
    outputPath = parentDir + '/Output/'
    _create_path(outputPath)
    
    #temp directory (e.g., training)
    tempPath = parentDir + '/Temp/'
    _create_path(tempPath)
    
    #output directory
    modelPath = parentDir + '/Model/'
    _create_path(modelPath)
    
    #state directory
    statePath = parentDir + '/State/'
    _create_path(statePath)
    
    # save encoder  
    affix = ".pkl"
    prename = 'encoder_'
    fencoder = _create_filename(modelPath, curFname, prename, affix)
    
    # save normalization scaler    
    prename = 'normScaler_'
    fnormScaler = _create_filename(modelPath, curFname, prename, affix)
    
    # save model   
    prename = 'clf_'
    fmodel = _create_filename(modelPath, curFname, prename, affix)
    
    # save status 
    affix = ".log"
    prename = 'run_'
    fstate = _create_filename(statePath, curFname, prename, affix)
    

    # <codecell> # model training or update
    if imode == 0:                 
        
        print('\n imode=0: Model training... \n')
        # select the model used for prediction in QA
        
        # read input data
        modelData = pd.read_csv(finput, error_bad_lines=False, thousands=',') #encoding='latin1') # Read the data              
          
        # import input tags
        modelData_inf = _replace_inf(modelData)
        X = modelData_inf[INPUT_TAGS]
        
        # read model data
        yTags = 'Claims'
        y = modelData_inf[yTags]
        
        # <codecell>
        
        print(modelData.shape, modelData.head())      
        
        # <codecell>
        """
        Preprocessing data
        
        """
        # split data into modeling and validation sets
        ts = 0.3
        xTrain, xTest, yTrain, yTest = train_test_split(X, y, stratify=y, test_size=ts, random_state=531)  
        
        # <codecell>
        # normalize data
        xvalue = xTrain.values 
        normX_scaler = preprocessing.StandardScaler().fit(xvalue) 
        #normX_scaler = preprocessing.MinMaxScaler().fit(xvalue) 
        
        xScaled = normX_scaler.transform(xvalue)   
        xScaled_test = normX_scaler.transform(xTest.values) 
        xnTrain = pd.DataFrame(xScaled)
        xnTest = pd.DataFrame(xScaled_test)
        
        xnTrain.index = xTrain.index
        xnTrain.columns = xTrain.columns
        xnTest.index = xTest.index
        xnTest.columns = xTest.columns
        
        #save scaler into file    
        with open(fnormScaler, "wb") as fo:
            pickle.dump(normX_scaler, fo)
        
        
        # <codecell>
        # class information
        _class_info(yTrain)
        _class_info(yTest)
        
        # <codecell>
        #split the training data into modeling set and validation set
        nxModel, nxValid, yModel, yValid = train_test_split(xnTrain, yTrain, stratify=yTrain, test_size=ts, random_state=531)
        
        # <codecell>
        _class_info(yModel)
        _class_info(yValid)
                
        # <codecell>
        """
        Modeling using scaled data
        """
        
        """
        # set modeling data for PCA        
        xnTrain = dpcaTrn
        nxValid = pcaValid
        xnTest = pcaTst
        
        # set modeling data for minMax normalized
        xnTrain = xmTrain
        nxModel = mxModel
        nxValid = mxValid
        xnTest = xmTest
        """
        # set modeling data 
        #nxModel = xnTrain
        #yModel = yTrain
        affix = ".pkl"  
        
        # predince model parameters
        xm = nxModel.iloc[:, 3:5]
        #xm = pd.DataFrame(xm)
        print(xm.head(2))
        n_features = X.shape[1]
        C = 1
        kernel = 1.0 * RBF(1.0)
        prior = [0.3, 0.7] # set the priors for classification
        test_results = ['Declined', 'Paid'] 
        
        # <codecell>
        # Create different classifiers. The logistic regression cannot do
        # multiclass out of the box.
        classifiers = {'L1_logistic': LogisticRegression(C=C, penalty='l1',
                                                         class_weight='balanced', solver = 'saga'),
                       'L2_logistic(OvR)': LogisticRegression(C=C, penalty='l2',
                                    class_weight='balanced', solver = 'saga'),
                       'L2_logistic(Multinomial)': LogisticRegression(C=C, solver='lbfgs', 
                                    multi_class='multinomial', class_weight='balanced'),               
                       'LinearSVC': SVC(kernel='linear', C=C, class_weight='balanced',
                                         probability=True, random_state=0),
                       'RBFSVC': SVC(kernel='rbf', C=10, gamma=2, class_weight='balanced',
                                      probability=True, random_state=0),
                       'SGD': SGDClassifier(loss= 'modified_huber', 
                                            class_weight='balanced',
                                            learning_rate = 'optimal',
                                            penalty="elasticnet"),
                       'NaiveBayes': GaussianNB(priors = prior),
                       'BGM': mixture.BayesianGaussianMixture(n_components = 10),                       
                       'NearestNeighbors': KNeighborsClassifier(4),
                       'DecisionTree': DecisionTreeClassifier(max_depth=5, class_weight = 'balanced'),
                       'EDT': ExtraTreesClassifier(n_estimators = 100,
                                                   max_depth=5, 
                                                   class_weight = 'balanced'),
                       'RandomForest': RandomForestClassifier(max_depth=20, n_estimators=200, 
                                                               max_features=1, class_weight = 'balanced'),
                       'NueralNetwork': MLPClassifier(alpha=1, activation = 'logistic', learning_rate = 'adaptive'),
                       'AdaBoost': AdaBoostClassifier(),
                       'GBRT': ensemble.GradientBoostingClassifier(n_estimators = 200,
                                                                   max_leaf_nodes = 3, 
                                                                   max_depth = 3,
                                                                   min_samples_split = 2,
                                                                   learning_rate = 0.1),
                       'QDA': QuadraticDiscriminantAnalysis(priors = prior),
                       'Bag': BaggingClassifier()
                       #'Blag': BlaggingClassifier() # for parallelling computing
                       #'GPC': GaussianProcessClassifier(kernel)  too time-consuming
                       }        
    
        n_classifiers = len(classifiers)   
        
        # <codecell>
        # classifiers comparison using full data sets
        rec_res = []
        clf_name = []
        for index, (name, classifier) in enumerate(classifiers.items()):
            print('\n Modeling...')
            classifier.fit(nxModel, yModel)
            print('\n Validation...')
            res_pred, rec_val = _verify_claim(classifier, name, nxValid, yValid,test_results)
            print('\n Testing...')
            res_test, rec_test = _verify_claim(classifier, name, xnTest, yTest,test_results)
            #save scaler into file    
            
            rec = pd.concat([pd.Series(rec_val), pd.Series(rec_test)], axis =0)
            recall = pd.DataFrame(rec, columns = [name])
            
            rec_res.append(recall[name].values)
            clf_name.append(name)
                                    
            fclf = _create_filename(modelPath, curFname, name, affix)
            with open(fclf, "wb") as fo:
                pickle.dump(classifier, fo)
        
        # plot probability curves
        _plot_prob(classifiers, nxModel, yModel, nxValid, yValid)
        
        rec_res = pd.DataFrame(rec_res)
        rec_resT = rec_res.transpose()
        rec_resT.columns = clf_name
        
        print(rec_resT)
        affix = ".csv"
        frecall = _create_filename(tempPath, curFname, 'recall', affix)
        with open(frecall, 'w') as csvfile:
            rec_resT.to_csv(csvfile)
            
        
        # <codecell>
        
        """
        Investigate voting methods for high-performance models
        """
        # C-Support Vector Classifier...0.30/0.72 - 0.31/0.72
        print("\n RBF-Support Vector Classifier...")
        
        # modeling
        #clf_svc = _plot_ROC_curve(SVC(C= 1, kernel = 'rbf', class_weight='balanced', 
        #                              probability = 1), nxModel, yModel)
        clf_svc = SVC(C=10, gamma=2, kernel = 'rbf', class_weight='balanced', probability = 1)
        clf_svc.fit(nxModel, yModel)
        name = 'RBF SVC'
        
        svc_output, svc_recall = _output_val_test(clf_svc, name, nxValid, yValid, xnTest, yTest,test_results)
        #svc_output.to_csv('svc_output.csv')
        print(svc_recall)

        ftempData = _create_filename(tempPath, curFname, name, affix)
        with open(ftempData, 'w') as csvfile:
            svc_output.to_csv(csvfile)
        
        
        # <codecell>
        
        # Random Forest Classifier...
        #print("\n Random Forest Classifier...")
        # modeling
        print("\n Random Forest Modeling...")
        #clf_rf = _plot_ROC_curve(RandomForestClassifier(max_depth=10, n_estimators=100,
        #                                                max_features=1, class_weight = 'balanced'), nxModel, yModel)
        clf_rf = RandomForestClassifier(max_depth=20, n_estimators=200,
                               max_features=1, class_weight = 'balanced')
        clf_rf.fit(nxModel, yModel)
        name = 'RF'
        
        rf_output, rf_recall = _output_val_test(clf_rf, name, nxValid, yValid, xnTest, yTest, test_results)
        print(rf_recall)
        #save output to temporary folder        
        ftempData = _create_filename(tempPath, curFname, name, affix)
        with open(ftempData, 'w') as csvfile:
            rf_output.to_csv(csvfile)  
        
       
        # <codecell>
        
        # modeling
        print("\n Decision Tree Modeling...")
        #clf_dt = _plot_ROC_curve(DecisionTreeClassifier(max_depth=10, class_weight = 'balanced'), nxModel, yModel)
        clf_dt = DecisionTreeClassifier(max_depth=5, class_weight = 'balanced')
        clf_dt.fit(nxModel, yModel)
        name = 'DT'
        
        dt_output, dt_recall = _output_val_test(clf_dt, name, nxValid, yValid, xnTest, yTest, test_results)
        #save output to temporary folder        
        ftempData = _create_filename(tempPath, curFname, name, affix)
        with open(ftempData, 'w') as csvfile:
            dt_output.to_csv(csvfile) 
        
        
        # <codecell>
        
        # modeling
        print("\n Bagging Classifier ...")
        clf_bag = BaggingClassifier().fit(nxModel, yModel)
        
        name = 'BAG'
        
        bag_output, bag_recall = _output_val_test(clf_bag, name, nxValid, yValid, xnTest, yTest, test_results)
        
        print(bag_recall)
        #save output to temporary folder        
        ftempData = _create_filename(tempPath, curFname, name, affix)
        with open(ftempData, 'w') as csvfile:
            bag_output.to_csv(csvfile)
        
        # <codecell>
        # be aware that it is very slow to create the learning curves
        """
        title = "Learning Curves (Naive Bayes)"
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        
        from sklearn.model_selection import ShuffleSplit
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
        
        estimator = GaussianNB()
        _plot_learning_curve(estimator, title, nxModel, yModel, ylim=(0.6, 0.75), cv=cv, n_jobs=4)
        
        title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
        # SVC is more expensive so we do a lower number of CV iterations:
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        estimator = SVC(gamma=0.01, C= 10, kernel = 'rbf', class_weight='balanced', probability = 1)
        _plot_learning_curve(estimator, title, nxModel, yModel, (0.5, 0.7), cv=cv, n_jobs=4)
        
        plt.show()
        """
        # <codecell>
        
        # modeling
        print("\n Voting Classifier Modeling...")
        eclf = VotingClassifier(estimators=[('dt', clf_dt), ('Bag', clf_bag), ('rf', clf_rf)], voting='soft', weights=[1,2,2])
        eclf = eclf.fit(nxModel, yModel)
        name = 'VCLF1'
        
        eclf_output, eclf_recall = _output_val_test(eclf, name, nxValid, yValid, xnTest, yTest, test_results)
        
        print(eclf_recall)
        #save output to temporary folder        
        ftempData = _create_filename(tempPath, curFname, name, affix)
        with open(ftempData, 'w') as csvfile:
            eclf_output.to_csv(csvfile)
            
        #save model into file 
        affix = ".pkl"
        fmodel = _create_filename(modelPath, curFname, name, affix)
        with open(fmodel, "wb") as fo:
            pickle.dump(eclf, fo)
           
        # <codecell>
        
        # modeling
        print("\n Voting Classifier Modeling...")
        aclf = VotingClassifier(estimators=[('svc', clf_svc), ('Bag', clf_bag), ('rf', clf_rf)], voting='soft', weights=[2,2,1])
        aclf = aclf.fit(nxModel, yModel)
        name = 'VCLF2'
        
        aclf_output, aclf_recall = _output_val_test(aclf, name, nxValid, yValid, xnTest, yTest, test_results)
        print(aclf_recall)
        
        #save output to temporary folder 
        affix = ".csv"
        ftempData = _create_filename(tempPath, curFname, name, affix)
        with open(ftempData, 'w') as csvfile:
            aclf_output.to_csv(csvfile)
            
        #save model into file 
        affix = ".pkl"
        fmodel = _create_filename(modelPath, curFname, name, affix)
        with open(fmodel, "wb") as fo:
            pickle.dump(aclf, fo)
                
        # <codecell>
        # Plotting decision tree
        feature_names=nxModel.columns.values
        class_names=test_results
        
        graph = _plot_decision_tree(clf_dt,feature_names, class_names)  
        graph 
        
        # <codecell>
        y_test = aclf_output['Claims']
        y_pred = aclf_output['Pred']
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        
        # Plot non-normalized confusion matrix
        plt.figure()
        _plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix, without normalization')
        
        # Plot normalized confusion matrix
        plt.figure()
        _plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')
        
        plt.show()
        
        # <codecell>
        
        # check which classifier is available for predict_proba...
        from sklearn.utils.testing import all_estimators
        
        estimators = all_estimators()
        
        for name, class_ in estimators:
            if hasattr(class_, 'predict_proba'):
                print(name)
    # <codecell>
    else: # prediction
        
        print('\n imode=1: Model predicting... \n')
        
        print('\n ... Extract data table from hana server')
        modelData = _pull_hana_hdb()
               
        if modelData.empty:
            with open(fstate, "w") as text_file:
                print("\n****************************", file=text_file)
                print("*...No any new claim now...*", file=text_file)
                print("****************************\n", file=text_file)    
                    
        else:
            with open(fstate, "w") as text_file:
                print("\n ... Extract data table from hana server", file=text_file)
                       
            # save data   
            affix = ".csv"
            prename = 'data_'
            now = datetime.datetime.now()
            fdata = _create_filename(dataPath, now.strftime("%Y-%m-%d"), prename, affix)
            
            with open (fdata, 'w') as csvfile:
                #csvwriter = csv.writer(csvfile, delimiter=',')
                #csvwriter.writenow(modelData)
                modelData.to_csv(csvfile)                
                        
        print('\n ... Data preprocessing ...')
        
        modelData.rename(columns={'TOTAL_CLAIM_CNT': 'numClaims', 'TRX_CNT': 'numSales', 
                                  'TOTAL_CLAIM':'dollarClaims', 'NET_SALES':'dollarSales'}, inplace=True)
        tagList = ['numClaims', 'numSales','dollarClaims', 'dollarSales']
        
        for item in tagList:
            modelData = _convert_float(modelData, item)
        
        modelData['dollarPerClaim'] = modelData['dollarClaims']/modelData['numClaims']
        modelData['dollarPerSale'] = modelData['dollarSales']/modelData['numSales']
        modelData['claimRatio'] = modelData['dollarClaims']/modelData['dollarSales']
        
        print(modelData.head(4), modelData.shape)
        
        # import input tags
        modelData_inf = _replace_inf(modelData)
        X = modelData_inf[INPUT_TAGS]
        
        # <codecell>
        print('\n ... product property encodeing...')
                
        # load encoder parameters created from modeling data
        with open(fencoder, 'rb') as fo:
            encoder_style, encoder_back, encoder_size, encoder_color = pickle.load(fo)
            
        #encodeing
        tag_style = 'INVENTORY_STYLE_CD'
        X, numb_style = _encode_label_(X, tag_style, encoder_style)
        
        tag_back = 'INVENTORY_BACKING_CD'
        X, numb_back = _encode_label_(X, tag_back,encoder_back)
        
        tag_size = 'INVENTORY_SIZE_CD'
        X, numb_size = _encode_label_(X, tag_size, encoder_size)
        
        tag_color = 'INVENTORY_COLOR_CD'
        X, numb_color = _encode_label_(X, tag_color, encoder_color)
        
        # <codecell>
        
        print('\n ... input data normalization ...')
        
        # load normalization scaler created from modeling data
        with open(fnormScaler, 'rb') as fo:
            XnormScaler = pickle.load(fo)
        
        xnTst = XnormScaler.transform(X.values)  # normalize
        xnTst = pd.DataFrame(xnTst)
        xnTst.index = X.index
        xnTst.columns = X.columns
        
        # <codecell>
        
        print('\n ... model prediction ...')
        
        # load trained clf model
        with open(fmodel,'rb') as fo:
            clf = pickle.load(fo)

        pred_output = _predict_claim(clf, xnTst)
        
        pred_op = pd.concat([modelData, pred_output], axis=1)
        
        #save results to the file
        affix = ".csv"
        prename = 'output_'        
        foutput = _create_filename(outputPath, now.strftime("%Y-%m-%d"), prename, affix)        
        with open (foutput, 'w') as csvfile:
            pred_op.to_csv(csvfile)               


# <codecell>
"""
if __name__ == '__main__':
    imode = 0
    for file_ in allFiles:    
        main(file_, imode) 
"""