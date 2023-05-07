import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score, accuracy_score


def rbf_svm():
    print('Generating results for Q3...')

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
    param_grid = {'C': np.logspace(-3, 3, 7), 'gamma': np.logspace(-3, 3, 7)}
    svm = SVC(kernel='rbf')
    grid_search = GridSearchCV(svm, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameter setting
    best_C = grid_search.best_params_['C']
    best_gamma = grid_search.best_params_['gamma']

    # Train the final RBF kernel SVM model on the entire training dataset with the best hyperparameter setting
    svm_final = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
    svm_final.fit(X_train, y_train)

    # Predict on the test dataset
    y_pred = svm_final.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)
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
  rbf_svm()
