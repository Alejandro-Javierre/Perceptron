import pandas as pd
import numpy as np
import random as rd

#   Randomness generation seed
np.random.seed(20)


def GetData(fileName):
    data = pd.read_csv(fileName, header=None, sep=",")
    data = data.to_numpy()
    return data


def _Predictor(inputValues, weights):
    #   Adds the bias to the activation score variable
    activationScore = weights[0]
    for i in range(0, len(inputValues)-1):
        #   Adds weight multiplied by input value to activation score
        activationScore += weights[i+1] * inputValues[i]
    #   returns 1 if the perceptron fires and 0 if not
    return 1 if activationScore >= 0 else -1


def BinaryPerceptron(data, classifier, exclusionClassifier="_Null",
                     threshold=1, regLambda=0, _multiClass=False):
    data = data.copy()
    weights = [0]
    epoch = 1
    trueScores = 0
    falseScores = 0
    classLabel = (len(data[0]) - 1)
    #   Cleans the data by removing excluded classifier (if present)
    data = np.delete(data, np.where(
        data[:, classLabel] == exclusionClassifier), axis=0)
    allClassifiers = np.unique(data[:, classLabel])
    #   Applies a mask to dataset for classifiers as positive and negative
    negativeClassMask = data[:, classLabel] != classifier
    data[:, classLabel][negativeClassMask] = -1
    positiveClassMask = data[:, classLabel] == classifier
    data[:, classLabel][positiveClassMask] = 1
    for i in range(0, classLabel):
        #   Creates four zero values for weights
        weights.append(0)
    #   Iterator to denote number of passes perceptron makes over training data
    while epoch <= threshold:
        #   shuffles the data after every pass through the dataset
        np.random.shuffle(data)
        for row in data:
            #   Calls Prediction function to gain activation prediction
            predicted = _Predictor(row, weights)
            #   If statements to denote the outcome of the prediction
            if predicted != row[classLabel]:
                #   Calculates the loss function difference for the bias
                weights[0] += row[classLabel]
                #   Cycles through all weights
                for i in range(0, classLabel):
                    #   Applies l2 regularization if specified
                    weights[i+1] -= 2*regLambda*weights[i+1]
                    #   Adds the loss function difference
                    weights[i+1] += row[classLabel]*row[i]
                #   Increments the false score counter
                falseScores += 1
            else:
                #   Increments true score counter for correct classifications
                trueScores += 1
        #   Increments epoch counter for new cycle of the dataset
        epoch += 1
    #   If statement to hide the accuracy report for multi class use
    if _multiClass is False:
        print("--------------------------------------------------------")
        print("Perceptron training performance:")
        print("Classifiers:", allClassifiers)
        print("accuracy:", (trueScores / (trueScores + falseScores)))
        print("--------------------------------------------------------")
    return weights


def PerceptronTest(data, weights, classifier, exclusionClassifier="_Null"):
    #   Initializes false and true test scores for accuracy report
    trueScores = 0
    falseScores = 0
    data = data.copy()
    #   Initializes the value of the final column (the class label column)
    classLabel = len(data[0]) - 1
    #   Deletes the excluded classifier if specified
    data = np.delete(data, np.where(
        data[:, classLabel] == exclusionClassifier), axis=0)
    #   Checks the number of classifiers for the final report
    allClassifiers = np.unique(data[:, classLabel])
    #   Converts classifier column into -1 or 1 depending on classifier label
    negativeClassMask = data[:, classLabel] != classifier
    data[:, classLabel][negativeClassMask] = -1
    positiveClassMask = data[:, classLabel] == classifier
    data[:, classLabel][positiveClassMask] = 1
    for row in data:
        #   Calls the Prediction function for an activation prediction
        predicted = _Predictor(row, weights)
        #   If statements to denote the outcome of the prediction
        if predicted == row[classLabel]:
            trueScores += 1
        else:
            falseScores += 1
    print("--------------------------------------------------------")
    print("Perceptron Test final output:")
    print("Classifiers:", allClassifiers)
    print("accuracy:", (trueScores/(trueScores + falseScores)))
    print("--------------------------------------------------------")


def MultiClassClassifier(trainData, testData, *classifiers,
                         threshold=1, l2Lambda=0):
    classifierWeights = []
    print("--------------------------------------------------------")
    print("Multi class classifier parameters:")
    print("Classifiers:", classifiers)
    print("Threshold:", threshold)
    print("L2 regularization Lambda:", l2Lambda)
    print("--------------------------------------------------------")
    #   Cycles through each classifier for multi class weight generation
    for classifier in classifiers:
        #   Adds results of a training algorithm to list of weights
        classifierWeights.append(BinaryPerceptron(
            trainData, classifier, threshold=threshold,
            regLambda=l2Lambda, _multiClass=True))
    #   Calls One V Rest test algorithm for both training and test datasets
    _OneVRestTest(trainData, "training", classifierWeights, *classifiers)
    _OneVRestTest(testData, "test", classifierWeights, *classifiers)
    return classifierWeights


def _OneVRestTest(data, datasetName, weights, *classifiers):
    data = data.copy()
    #   Initializes true and false scores for final accuracy report
    trueScores = 0
    falseScores = 0
    classCounter = 1
    #   Initializes the value of the final column (the class label column)
    classLabel = len(data[0]) - 1
    #   Cycles through classifier weight lists
    for classifier in classifiers:
        #   Provides mask to denote class number
        classMask = data[:, classLabel] == classifier
        data[:, classLabel][classMask] = classCounter
        classCounter += 1
    #   Iterates through each row of the dataset
    for row in data:
        #   Initalizes list to collect activation scores
        argmaxScore = []
        for classifier in weights:
            #   sets up bias value for classifier
            activationScore = classifier[0]
            #   Iterates through each classifier weight
            for i in range(0, classLabel):
                #   Activation score function
                activationScore += classifier[i+1] * row[i]
            #   Adds activation score to list of other scores
            argmaxScore.append(activationScore)
        #   Chooses the index of the largest value (+1 to get the label)
        classPrediction = (argmaxScore.index(max(argmaxScore)) + 1)
        #   If statement to check if prediction is correct
        if classPrediction == row[classLabel]:
            trueScores += 1
        else:
            falseScores += 1
    print("--------------------------------------------------------")
    print("One Vs Rest Test final output:")
    print("dataset: ", datasetName)
    print("accuracy:", (trueScores/(trueScores + falseScores)))
    print("--------------------------------------------------------")

print("GETTING DATASETS")

train_data = GetData("train - Copy.data")

test_data = GetData("test-Copy1.data")

print("3. BINARY PERCEPTRON")

print("a. class 1 and class 2")

Class1and2Weights = BinaryPerceptron(train_data,
                                     "class-1", "class-3", threshold=20)

PerceptronTest(test_data, Class1and2Weights, "class-1", "class-3")

print("b. class 2 and class 3")

Class2and3Weights = BinaryPerceptron(train_data,
                                     "class-2", "class-1", threshold=20)

PerceptronTest(test_data, Class2and3Weights, "class-2", "class-1")

print("c. class 1 and class 3")

Class1and3Weights = BinaryPerceptron(train_data,
                                     "class-1", "class-2", threshold=20)

PerceptronTest(test_data, Class1and3Weights, "class-1", "class-2")

print("4. MULTICLASS TEST")

finalWeights = MultiClassClassifier(train_data, test_data,
                                    "class-1", "class-2", "class-3",
                                    threshold=20)

print("5. L2 REGULARIZATION")

MultiClassClassifier(train_data, test_data,
                     "class-1", "class-2", "class-3",
                     threshold=20, l2Lambda=0.01)

MultiClassClassifier(train_data, test_data,
                     "class-1", "class-2", "class-3",
                     threshold=20, l2Lambda=0.1)


MultiClassClassifier(train_data, test_data,
                     "class-1", "class-2", "class-3",
                     threshold=20, l2Lambda=1)

MultiClassClassifier(train_data, test_data,
                     "class-1", "class-2", "class-3",
                     threshold=20, l2Lambda=10)

MultiClassClassifier(train_data, test_data,
                     "class-1", "class-2", "class-3",
                     threshold=20, l2Lambda=100)
