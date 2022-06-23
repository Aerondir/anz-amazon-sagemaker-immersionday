# Python Built-Ins:
import argparse
import os

# External Dependencies:
#Joblib is a set of tools to provide lightweight pipelining in Python.
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, 
# along with a large collection of high-level mathematical functions to operate on these arrays.

#  pandas is a software library written for the Python programming language for data manipulation and analysis. 
#In particular, it offers data structures and operations for manipulating numerical tables and time series.

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#A random forest is a meta estimator that fits a number of classifying decision trees 
#on various sub-samples of the dataset and uses averaging to improve the predictive accuracy 
#and control over-fitting.

# inference functions ---------------
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


if __name__ == '__main__':

    #------------------------------- parsing input parameters (from command line)
    print('extracting arguments')
    parser = argparse.ArgumentParser()

    # RandomForest hyperparameters
    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--min_samples_leaf', type=int, default=3)

    # Data, model, and output directories
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test_dir', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train_file', type=str, default='california_housing_train.csv')
    parser.add_argument('--test_file', type=str, default='california_housing_test.csv')
    parser.add_argument('--features', type=str)  # explicitly name which features to use
    parser.add_argument('--target_variable', type=str)  # explicitly name the column to be used as target

    args, _ = parser.parse_known_args()

    #------------------------------- data preparation
    print('reading data')
    train_df = pd.read_csv(os.path.join(args.train_dir, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test_dir, args.test_file))

    print('building training and testing datasets')
    X_train = train_df[args.features.split()]
    X_test = test_df[args.features.split()]
    y_train = train_df[args.target_variable]
    y_test = test_df[args.target_variable]

    #------------------------------- model training
    print('training model')
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=-1)

    model.fit(X_train, y_train)

    #-------------------------------  model testing
    print('testing model')
    abs_err = np.abs(model.predict(X_test) - y_test)

    # percentile absolute errors
    for q in [10, 50, 90]:
        print('AE-at-' + str(q) + 'th-percentile: '
              + str(np.percentile(a=abs_err, q=q)))
        
#Mean Absolute Error is a model evaluation metric used with regression models. 

#------------------------------- save model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print('model saved at ' + path)
