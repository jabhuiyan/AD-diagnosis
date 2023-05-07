import pandas as pd
from sklearn.svm import SVC


def diagnoseDAT(Xtest, data_dir):
    """Returns a vector of predictions with elements "0" for sNC and "1" for sDAT,
    corresponding to each of the N_test features vectors in Xtest.
    
    Xtest: N_test x 14 matrix of test feature vectors
    data_dir: full path to the folder containing the following files:
    train.fdg_pet.sNC.csv, train.fdg_pet.sDAT.csv,
    test.fdg_pet.sNC.csv, test.fdg_pet.sDAT.csv
    """

    # Load the required datasets
    train_NC = pd.read_csv(data_dir + "/train.fdg_pet.sNC.csv", header=None)
    train_DAT = pd.read_csv(data_dir + "/train.fdg_pet.sDAT.csv", header=None)

    # Combine the two training datasets and add labels
    train_NC["label"] = 0
    train_DAT["label"] = 1
    train_data = pd.concat([train_NC, train_DAT])

    # Split the training data into features (X) and labels (y)
    X_train = train_data.drop("label", axis=1)
    y_train = train_data["label"]

    # Train the "best" SVM model using C =1, degree = 3
    svm = SVC(kernel='poly', C=1, degree=3)
    svm.fit(X_train, y_train)

    return svm.predict(Xtest)

