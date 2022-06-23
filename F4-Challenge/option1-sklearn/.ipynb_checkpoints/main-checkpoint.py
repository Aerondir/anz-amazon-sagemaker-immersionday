"""
File: BYO_scikitlearn_model
"""
#import all the libraries you need
import argparse
import numpy as np
import os
import pandas as pd
from sklearn.externals import joblib
#TODO add your other required libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#functions used in your code


# -TODO-Dictionary to encode labels to codes

# Dictionary to encode labels to codes
label_encode = {
    'Iris-virginica': 0,
    'Iris-versicolor': 1,
    'Iris-setosa': 2
}


# Dictionary to convert codes to labels

# Dictionary to convert codes to labels
label_decode = {
    0: 'Iris-virginica',
    1: 'Iris-versicolor',
    2: 'Iris-setosa'
}


#training script
if __name__ =='__main__':
     #------------------------------- parsing input parameters (from command line)
    print('extracting arguments')
    parser = argparse.ArgumentParser()
       
   #TODO: RandomForest hyperparameters
    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--min_samples_leaf', type=int, default=3)


   # TODO: Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    
        
    args = parser.parse_args()
    

    #TODO-Load your data (both training and test) from container filesystem into training / test data sets and split them to train/test features and lables
    train = pd.read_csv(os.path.join(args.train,'train.csv'),index_col=0, engine="python")
    y_train= train['label'].map(label_encode)
    X_train =  train.drop(["label"], axis=1)

    test = pd.read_csv(os.path.join(args.test,'test.csv'),index_col=0, engine="python")
    y_test= test['label'].map(label_encode)
    X_test =  test.drop(["label"], axis=1)

     
    #TODO- fit the randomforest model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf)

    model.fit(X_train, y_train)   

    #Save the model to the location specified by args.model_dir, using the joblib
    
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print('model saved at ' + path)

    
    
    
#define a function that loads the model    
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


#define a custom output function that takes the prediction and changes the numeric labels to the string lables (bonus!)

def output_fn(prediction, content_type):
    return ' | '.join([label_decode[t] for t in prediction])