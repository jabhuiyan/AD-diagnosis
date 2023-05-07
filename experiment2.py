import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score, accuracy_score

def polynomial_svm():
    # Load training data
    train_sNC = pd.read_csv("train.fdg_pet.sNC.csv", header=None)
    train_sDAT = pd.read_csv("train.fdg_pet.sDAT.csv", header=None)

    train_sNC["label"] = 0
    train_sDAT["label"] = 1
    train_data = pd.concat([train_sNC, train_sDAT])

    X_train = train_data.drop("label", axis=1)
    y_train = train_data["label"]

    # Load test data
    test_sNC = pd.read_csv("test.fdg_pet.sNC.csv", header=None)
    test_sDAT = pd.read_csv("test.fdg_pet.sDAT.csv", header=None)

    test_sNC["label"] = 0
    test_sDAT["label"] = 1
    test_data = pd.concat([test_sNC, test_sDAT])

    X_test = test_data.drop("label", axis=1)
    y_test = test_data["label"]

    # Perform hyperparameter tuning using GridSearchCV
    # List of values to explore for regularization parameter C and degree of polynomial kernel
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 15], 'degree': [2, 3]}
    svm = SVC(kernel='poly')  # Polynomial kernel SVM model
    # 5-fold cross-validation
    grid_search = GridSearchCV(svm, param_grid, cv=5)
    grid_search.fit(X_train, y_train)  # Fit the model

    # Extract the best hyperparameter setting
    best_C = grid_search.best_params_['C']
    best_degree = grid_search.best_params_['degree']

    print('Best C: ', best_C, ' and best deg: ', best_degree)

    # Train the final model using the best hyperparameter setting on the entire training dataset
    svm_final = SVC(kernel='poly', C=best_C, degree=best_degree)
    svm_final.fit(X_train, y_train)

    # Predict on the test dataset
    y_pred = svm_final.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    # Print performance metrics
    print("Accuracy: {:.4f}".format(accuracy))
    print("Sensitivity: {:.4f}".format(sensitivity))
    print("Specificity: {:.4f}".format(specificity))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("Balanced Accuracy: {:.4f}".format(balanced_accuracy))

if __name__ == "__main__":
  polynomial_svm()
  
