import os, sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
import pandas as pd
import scipy
import sklearn
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder

# Open data and define columns
data = pd.read_csv('dac_sample.txt', sep="\t", header = None,
                   names = ["Label", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13",
                 "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13",
                "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"])

np.sum(data.isnull())

#
data = pd.DataFrame({col: data[col].astype('category').cat.codes for col in data}, index = data.index)

data[['I1', 'I2', "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13"]] = \
    data[["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13"]].apply(lambda x: pd.to_numeric(x, errors='force'))

# Fill in with missing data
data.fillna(data.mean())

# Prepare predictor variables for training/testing sets
predictors = data.drop('Label', 1)

# Prepare response variables for training/testing sets
response = data['Label']

## Logistic Regression
from sklearn.linear_model import LogisticRegression
# Create training/testing samples for predictors/response variables
predictors_train, predictors_test, response_train, response_test = cross_validation.train_test_split(
    predictors, response, test_size=0.4, random_state=0)
# Create regression object and Train the model using the training sets
lr = LogisticRegression().fit(predictors_train, response_train)
# The coefficients
print('Coefficients: \n', lr.coef_)
# The intercept
print('Coefficients: \n', lr.intercept_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((lr.predict(predictors_test) - response_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lr.score(predictors_test, response_test))
# Cross-Validation Scores
lr.score(predictors_test, response_test)

## Stochastic Gradient Descent
from sklearn import linear_model
# Create training/testing samples for predictors/response variables
predictors_train, predictors_test, response_train, response_test = cross_validation.train_test_split(
    predictors, response, test_size=0.4, random_state=0)
# Create regression object and Train the model using the training sets
sgd = linear_model.SGDClassifier (alpha = .5).fit (predictors_train, response_train)
# The coefficients
print('Coefficients: \n', sgd.coef_)
# The intercept
print('Coefficients: \n', sgd.intercept_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((sgd.predict(predictors_test) - response_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % sgd.score(predictors_test, response_test))
# Cross-Validation Scores
sgd.score(predictors_test, response_test)

## Multi-layer Perceptron Classifier
from sklearn.neural_network import MLPClassifier
# Create training/testing samples for predictors/response variables
predictors_train, predictors_test, response_train, response_test = cross_validation.train_test_split(
    predictors, response, test_size=0.4, random_state=0)
# Create regression object and Train the model using the training sets
mlp = MLPClassifier(alpha = 0.1).fit (predictors_train, response_train)
# The coefficients
print('Coefficients: \n', mlp.coef_)
# The intercept
print('Coefficients: \n', mlp.intercept_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((mlp.predict(predictors_test) - response_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % mlp.score(predictors_test, response_test))
# Cross-Validation Scores
mlp.score(predictors_test, response_test)

## SVM Classification
from sklearn import svm
# Create training/testing samples for predictors/response variables
predictors_train, predictors_test, response_train, response_test = cross_validation.train_test_split(
    predictors, response, test_size=0.4, random_state=0)
# Create SVM object and Train the model using the training sets
svc = svm.SVC().fit (predictors_train, response_train)
# The coefficients
# print('Coefficients: \n', svr.coef_)
# The intercept
# print('Coefficients: \n', svr.intercept_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((svc.predict(predictors_test) - response_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % svc.score(predictors_test, response_test))
# Cross-Validation Scores
svc.score(predictors_test, response_test)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
# Create training/testing samples for predictors/response variables
predictors_train, predictors_test, response_train, response_test = cross_validation.train_test_split(
    predictors, response, test_size=0.4, random_state=0)
# Create GNB object and Train the model using the training sets
gnb = GaussianNB().fit (predictors_train, response_train)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((gnb.predict(predictors_test) - response_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % gnb.score(predictors_test, response_test))
# Cross-Validation Scores
gnb.score(predictors_test, response_test)

y_pred = gnb.fit(predictors_train, response_train).predict(predictors_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (predictors_test.shape[0],(response_test != y_pred).sum()))

## Random Forest
from sklearn.ensemble import RandomForestClassifier
# Create training/testing samples for predictors/response variables
predictors_train, predictors_test, response_train, response_test = cross_validation.train_test_split(
    predictors, response, test_size=0.4, random_state=0)
# Create RandomForest object and Train the model using the training sets
rfr = RandomForestClassifier(n_estimators=128, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                    max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1,
                            random_state=None, verbose=0, warm_start=False).fit (predictors_train, response_train)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((rfr.predict(predictors_test) - response_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % rfr.score(predictors_test, response_test))
# Cross-Validation Scores
rfr.score(predictors_test, response_test)

## Neural Networks
from sklearn.neural_network import BernoulliRBM
# Create training/testing samples for predictors/response variables
predictors_train, predictors_test, response_train, response_test = cross_validation.train_test_split(
    predictors, response, test_size=0.4, random_state=0)
# Create Neural Network object and Train the model using the training sets
rbm = BernoulliRBM(random_state=0, verbose=True)
logistic = linear_model.LogisticRegression()
logistic.fit(predictors_train, response_train)
classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)]).fit(predictors_train, response_train)
# Get predictions
print("The RBM model:")
print ("Predict:"), classifier.predict(predictors_test)
print ("Real:,", response_test)
print