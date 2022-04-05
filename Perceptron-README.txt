PERCEPTRON ALGORITHM
Alejandro Javierre

------------------------------------------
Setup:

The algorithm requires a training and testing data file.

The perceptron is split into five functions: GetData, _Predictor, BinaryPerceptron,
PerceptronTest, _OneVsRestTest and MultiClassClassifier.

NOTE: In order to allow for slight reproducibility despite randomized dataset shuffling 
and weight generation, I have included the randomly generated seed number of 69. In order 
to fully randomize the algorithm remove the seed generator under the module imports.
-----------------------------------------
Function inputs and outputs:

GetData is used to generate a compatible numpy dataset from most data files. 
	
	The only input for this function is the name of the file to be 
	extracted as a string. Include the .file suffix. It must be included
	in the same directory of the perceptron.py file. 

	It will output a numpy array corresponding to the data file, with each feature
	and the class being a column and each instance corresponds to a row.

BinaryPerceptron is used to train perceptron weights, and returns a list 
containing the resulting weights of the Perceptron learning process.This 
function has two required inputs and three optional inputs. If ran without
specifying the excluded class, it will not remove any classes and run a
one Vs rest classification perceptron.

	data: The dataset to train the perceptron on, as a numpy nd array
	
	classifier: The classifier the perceptron treats as a positive value, as a string.
	
	exclusionClassifier: A single value which denotes the classifier to be 
	excluded in the case of a dataset which includes three classifiers, as a string.
		(default: "_Null")

	threshold: The number of passes the Perceptron will run over the training dataset.
		(default: 1)
	
	regLambda: The lambda score for L2 Regularization, as a positive float or integer 
		(default: 0)

PerceptronTest is used to test the weights produced from the BinaryPerceptron function

	This function has three required inputs and one optional input.
	
	data: The dataset to test the perceptron on, as a numpy nd array

	weights: The set of weights created from the BinaryPerceptron function, as a list
		of values.

	classifier: The classifier the perceptron treats as a positive value, written as
		a string.

	exclusionClassifier: A single value which denotes the classifier to be 
	excluded in the case of a dataset which includes three classifiers.
		The exclusionClassifier will delete rows which are labeled
		as the classifier
		(default: "_Null")

MultiClassClassifier is a function with two required parameters, and a variable number of
	optional parameters. The function should be called with arguments in this order:
	
	trainData: The training set for the one Vs rest perceptrons, as a numpy nd array
	
	testData: The final testing data for the multiclassification, as a numpy, nd array

	classifiers: A variable length argument providing the exact names of each 
		classifier to be trained in the datasets as strings.
	
	classThreshold: The number of epochs to train each perceptron with, as a positive integer
		WARNING: To change the threshold value, the argument must be specified
		after listing classifiers.
		(default: 1)

	l2Lambda: The lambda score for L2 Regularization, as a positive float or integer 
		WARNING: To change the L2 Lambda value, the argument must be specified
		after listing classifiers.
		(default: 0)
	
	An example of how the MultiClassClassifier function should be called can be
	found below:

	MultiClassClassifier(trainData, testData, 20, 0.001, "class-1", "class-2", "class-3")

	The function will return a two-dimensional list containing the final weights for
	each class, in the order they are called in the function.
		At index 0 for each class weight list is the bias, with subsequent
		values corresponding to weights for features in the dataset.	

_Predictor:
	This function is called automatically and should not be used outside of
	the BinaryPerceptron function.

_OneVRestTest:
	This function is called automatically and should not be used outside of
	the MultiClassClassifier function.

-----------------------------------------
How to get data for, train and test a binary perceptron algorithm

Step 1:
	Use the GetData() function to generate a training and test set and save 
	to two clearly labelled variables, using the name of the target files saved
	to the same workspace as the perceptron algorithm. An example can be seen below:

		trainData = GetData("training_values.data")
	
		testData = GetData("test_values.data")
Step2:
	use the BinaryPerceptron() function to use the train data to create a set of
	weights. The classifier and exclusionClassifier variable must be called after
	specifying the dataset, with the threshold number and L2 regularization, if used,
	being specified afterwards. An example can be seen below:

		weights = BinaryPerceptron(trainData, "2", "3", threshold=20)
	
	This will train a set of perceptron weights from the trainData dataset to distinguish		
	between classes with the label of "2" and "1", and excluding instances which are
	labelled as "3".
Step 3:
	Test the perceptron weights using the PerceptronTest() function. This requires a
	test dataset, training weights and the classifier and exclusion classifier to be
	specified again. For the example above, a test for this algorithm would be:
	
		PerceptronTest(testData, weights, "2", "3")
-----------------------------------------
How to run a one VS rest multiclassification on a three-classifier dataset

Step 1:

	Use the GetData() function to generate a training and test set and save 
	to two clearly labelled variables, using the name of the target files saved
	to the same workspace as the perceptron algorithm. An example can be seen below:

		trainData = GetData("training_values.data")
	
		testData = GetData("test_values.data")

Step 2:

	use the MultiClassClassifier function by inputting the training and test data, 
	all of the classifiers to be distinguished in the dataset, as well as the threshold 
	and L2 Lambda values (if different to the default). See example below:

		classWeights = MultiClassClassifier(trainData, testData, 
							"class-1", "class-2", "class-3")
		
	optionally if threshold and lambda values are different to the default:

		classWeights = MultiClassClassifier(trainData, testData, 
						"class-1", "class-2", "class-3",
							threshold=20, l2Lambda=0.01)
Step 3: 

	The MultiClassClassifier function should report the accuracies of the perceptron
	training algorithms as well as the accuracy of the one VS rest test. The function
	will output a two-dimensional list containing the bias and weights of each class.