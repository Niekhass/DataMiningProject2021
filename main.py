# Author: Niek Hasselerharm (s1026769)
# Created as part of a project for the course NWI-IBI008 Data Mining.
#
# An implementation of the VFDT described in the paper "Mining High-Speed Data Streams" by Domingos & Hulten.
# See: http://web.cs.wpi.edu/~cs525/f13b-EAR/cs525-homepage/lectures/PAPERS/p71-domingos.pdf


# Python standard library imports
from time import time

# Other library imports
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
from pandas import read_csv
import numpy as np

# Project-file imports
from HoeffdingTree import HoeffdingTree


# Function to handle the extraction and pre-processing of data from an input file.
def pre_processing(infile):
    attributes = None
    X = None
    y = None

    # CSV handling
    if infile.endswith(".csv"):
        # Read CSV file and extract data.
        df = read_csv(infile)
        data = df.values

        # Extract table X, classes y and the encoded list of attributes.
        X = data[:, :-1]
        y = data[:, -1]
        attributes = list(range(X.shape[1]))

    # Matlab handling
    elif infile.endswith(".mat"):
        # Read Matlab file and extract data
        data = loadmat(infile)

        # Extract table X, classes y and the encoded list of attributes.
        X = data.get("X")
        y = data.get("y").ravel()
        attributes = list(range(X.shape[1]))

    # Shuffle samples randomly.
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    X = X[index]
    y = y[index]

    # Encode discrete attributes as continuous values.
    encoding = {}
    for i in range(len(X)):
        for j in range(len(X[i])):
            # Check if the value is discrete.
            if not isinstance(X[i][j], (int, float, complex)):
                attr = encoding.get(j)
                # Check if the attribute has an entry.
                if attr is None:
                    # If not, create an entry and add encoding for the current value.
                    encoding.update({j: {}})
                    encoding[j].update({X[i][j]: 0})
                else:
                    # Otherwise, check if this value has an encoding.
                    val = encoding[j].get(X[i][j])
                    if val is None:
                        # If not, add a new encoding for this value.
                        encoding[j].update({X[i][j]: len(encoding[j])})

                # Lastly, set the value of the discrete attribute to the continuous encoding.
                X[i][j] = encoding[j][X[i][j]]

    return X, y, attributes


# Executes the VFDT algorithm with the provided input file.
def run(infile):
    # Obtain the sample data table X, corresponding classes y and list of attributes.
    X, y, attributes = pre_processing(infile)
    n_min = 600         # Minimum number of samples a node has to have processed before it can split.
    d = 0.01            # The delta used in the computation of the Hoeffding bound.
    t = 0.05            # The user-specified tau threshold used in the ([difference in G] < epsilon < tau) check in case of a "tie".
    train_split = 0.8   # The portion of total samples to be used for training. The remainder will be used for testing.

    # Compute split and divide data into train- and testing data.
    n_train = int(X.shape[0] * train_split)
    print("Data set contains {} samples. \
          \nTrain data: {} samples ({:.0f}%)\
          \nTest data: {} samples ({:.0f}%)\n".format(X.shape[0], n_train, train_split * 100, X.shape[0] - n_train, 100 - train_split * 100))

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_test = X[n_train:X.shape[0]]
    y_test = y[n_train:X.shape[0]]

    attributes = list(range(X.shape[1]))

    ht = HoeffdingTree(n_min, d, t, attributes)

    start_time = time()

    # Train the tree using test data.
    for sample in zip(X_train, y_train):
        ht.train(sample[0], sample[1])

    train_end_time = time()

    print("Total training time: {:.4f} s".format(train_end_time - start_time))
    print("Average time per sample: {:.4f} s\n".format((train_end_time - start_time)/X.shape[0]))

    # Perform testing and measure accuracy.
    y_pred = []
    for x in X_test:
        y_pred.append(ht.predict(x))

    end_time = time()

    acc = accuracy_score(y_test, y_pred) * 100
    run_time = end_time - start_time

    print("Achieved accuracy: {:.2f}%".format(acc))
    print("Total running time: {:.4f} s".format(run_time))

    return acc, run_time


def main():
    # List of input files used for testing.
    infiles = ["wine.mat", "phplE7q6h.csv"]
    infile = infiles[1]

    scores = []
    run_times = []
    for i in range(0, 5):
        acc, run_time = run(infile)
        scores.append(acc)
        run_times.append(run_time)

    print("Accuracies over 5 runs: ", scores)
    print("Runtimes over 5 runs:", run_times)


if __name__ == "__main__":
    main()
