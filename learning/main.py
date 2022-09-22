import getopt
import sys

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.svm import LinearSVC

VERSION = "0.0.1"


def parameter_optimizer(params, x_train, y_train):
    print_message("Searching and training Model to the bests Parameters (can be long)")
    m = LinearSVC(multi_class='ovr', C=1, max_iter=2000)
    clf = GridSearchCV(m, params)
    clf.fit(x_train, y_train)
    print(clf.best_estimator_)
    return clf.best_estimator_


def print_message(message: str, stream: sys = sys.stdout):
    stream.write("{}\n".format(message))


def main(argc: int, argv: list):
    usage_message = "Usage {} [-v] -i <features_csv> -o <export_json>".format(argv[0])
    verbose = False
    features_file = ""
    model_export_file = ""
    print("AU Features Learner v{}".format(VERSION))
    if argc == 1:
        print_message(usage_message, sys.stderr)
        exit(1)
    try:
        opts, args = getopt.getopt(argv[1:], "vo:i:h")
    except getopt.GetoptError as err:
        print_message(str(err), sys.stderr)
        exit(1)

    for options, argument in opts:
        if options == "-h":
            print_message(usage_message, sys.stdout)
            exit(0)
        elif options == "-i":
            features_file = argument
        elif options == "-o":
            model_export_file = argument
        elif options == "-v":
            verbose = True
        else:
            print_message(usage_message, sys.stdout)
            exit(1)
    if features_file == "" or model_export_file == "":
        print_message(usage_message, sys.stderr)
        exit(1)
    learn_data(features_file, model_export_file, verbose)


def learn_data(input_path, export_path, verbose):
    dataset_df = pd.read_csv(input_path, sep=",")

    print("Learning W/ Features from: \"{}\"".format(input_path))
    print("Exporting Model to: \"{}\"".format(export_path))

    if verbose:
        print("Data Set Shape: {}".format(dataset_df.shape))

    batch_audio = pd.DataFrame(dataset_df).to_numpy()

    y_name = batch_audio[:, -1]

    y = LabelEncoder().fit_transform(y_name)
    class_correspond = {}
    for i in range(len(y_name)):
        class_correspond[int(y[i])] = y_name[i]
    y_unique = np.unique(y_name)
    if verbose:
        print(y_unique)

    batch_val = batch_audio[:, :-1].astype('float32')

    batch_avg = np.average(batch_val, 0)
    batch_stddev = np.std(batch_val, 0)

    batch_normalized = (batch_val.copy() - batch_avg) / batch_stddev

    X_train, X_test, y_train, y_test = train_test_split(batch_normalized, y, test_size=0.25,
                                                        random_state=0)

    if verbose:
        print("X_train :", X_train.shape)
        print("y_train :", y_train.shape)
        print("X_test :", X_test.shape)
        print("y_test :", y_test.shape)
        print("Number of class tested:", np.unique(y_train))

    params = {'C': [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5], 'max_iter': [1500, 2000]}

    model = parameter_optimizer(params, X_train, y_train)
    model.fit(X_train, y_train)
    print("Model Accuracy: {}%".format(model.score(X_test, y_test) * 100))
    plot_confusion_matrix(model, X_test, y_test, normalize='true', display_labels=y_unique)

    model_param = {
        "intercept": np.array(model.intercept_).tolist(),
        "coef": np.array(model.coef_).tolist(),
        "normalisation": {
            "average": batch_avg.tolist(),
            "std": batch_stddev.tolist()
        },
        'classes': class_correspond
    }
    out_file = open(export_path, "w")
    out_file.write(json.dumps(model_param, indent=4))
    out_file.close()
    plt.show()


if __name__ == "__main__":
    main(len(sys.argv), sys.argv.copy())
