#A Basic Neural Network

###Introduction

This implements a very basic neural network, which can use a variety of activation functions (relu, leaky relu, sigmoid, and tanh), cost functions (mean squared error and mean hypercubed error).  When inputing an array which has an integer square root, you can have sparse connections from the input layer to the next layer, although this requires that the next layer be the same size.

###Syntax

Here's the basic idea:

	var i = new NN();
	i.set('layers',[5,5,2]); //The first of these is how big the input array should be, the last of these how big the output is.
	i.set('link','relu');	//Set the activation function
	i.set('randomness','proportionateZeroCentered'); //What kind of randomness do we use for the weights and bias initialization?
	i.init();	//Make initial neurons, in accord with the settings above.
	i.trainBatch(inputArrays, outPutArrays); //Train a batch on the above.
	i.score(validationInput, validationOutput);	//Rate
	i.predict(inputArrays);	//Outputs array of arrays

The output arrays can be any length and any value, although you'd be advised to use values > 1 only for relu and leaky relus, and value < 0 only for tanh.

The input arrays are simply arrays, which you'd want to scale similarly to the above, in accord with the activation function.

