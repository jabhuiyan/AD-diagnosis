import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tkinter as tk

def linear_svm():
    # Load the training data
    train_sNC = pd.read_csv("train.fdg_pet.sNC.csv", header=None)
    train_sDAT = pd.read_csv("train.fdg_pet.sDAT.csv", header=None)

    # Concatenate the training data and create labels
    X_train = pd.concat([train_sNC, train_sDAT])
    y_train = np.concatenate(
        [np.zeros(len(train_sNC)), np.ones(len(train_sDAT))])

    # Load the test data
    test_sNC = pd.read_csv("test.fdg_pet.sNC.csv", header=None)
    test_sDAT = pd.read_csv("test.fdg_pet.sDAT.csv", header=None)

    # Concatenate the test data and create labels
    X_test = pd.concat([test_sNC, test_sDAT])
    y_test = np.concatenate([np.zeros(len(test_sNC)), np.ones(len(test_sDAT))])

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Define the parameter grid to search over
    param_grid = {'C': [0.1, 1, 10, 100, 1000]}

    # Create a SVM classifier object
    clf = SVC(kernel='linear')

    # Create a GridSearchCV object and fit it to the training data
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameter setting
    print("Best hyperparameter setting: ", grid_search.best_params_)

    # Plot the performance of the models explored during the C hyperparameter tuning phase
    C_values = [0.1, 1, 10, 100, 1000]
    mean_scores = grid_search.cv_results_['mean_test_score']
    std_scores = grid_search.cv_results_['std_test_score']
    plt.plot(C_values, mean_scores, marker='o')
    plt.fill_between(C_values, mean_scores - std_scores,
                     mean_scores + std_scores, alpha=0.2)
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Mean cross-validated accuracy')
    plt.title('Performance of linear SVM as a function of C')
    plt.show()

    # Create and train the final linear SVM classifier with the best hyperparameter setting
    clf = SVC(kernel='linear', C=grid_search.best_params_['C'])
    clf.fit(X_train, y_train)

    # Evaluate the final model on the test data
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Calculate the sensitivity, specificity, precision, recall, and balanced accuracy
    tn, fp, fn, tp = confusion.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = sensitivity
    balanced_accuracy = (sensitivity + specificity) / 2

    # Print the performance metrics of the final model
    print("Accuracy:", accuracy)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Balanced Accuracy:", balanced_accuracy)
    print("Confusion matrix:\n", confusion)
    print("Classification report:\n", report)

    # Create a new window to display the performance metrics
    root = tk.Tk()
    root.geometry("500x300")
    root.title("Performance Metrics")

    # Create a label to display the performance metrics
    metrics_label = tk.Label(root, text="Accuracy: {}\nSensitivity: {}\nSpecificity: {}\nPrecision: {}\nRecall: {}\nBalanced Accuracy: {}\nConfusion Matrix:\n{}".format(
        accuracy, sensitivity, specificity, precision, recall, balanced_accuracy, confusion))
    metrics_label.pack()

    # Show the window
    root.mainloop()
 
if __name__ == "__main__":
  linear_svm()
