# Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch).

The neural network should be trained on the Training Set using stochastic gradient descent. It should achieve 97-98% accuracy on the Test Set.

## Solutions

a.	The Test Accuracy is around 0.9805 on the test data as code shown, after training the model with 20 iterations on training data. 


b.	The whole code is following the sequence of neural network implementation. 



-	Process with data: Load data -> normalization -> convert_to_onehot ( convert label Y )
-	Initialize Parameters: initial_par (with 120 hidden units, the input layer is 784 and the output layer is 10)
-	Feed Data Froward: Linear -> sigmoid(activate it to non-linear) -> Linear -> Softmax; also record the output of each layer in cache
-	Back Propagation: take gradient of Softmax -> Linear -> sigmoid -> Linear; also record gradients of each layer in grads
-	Update Parameters: update_par (update each parameter with different learning rate accounting to different iteration in each linear layer). Then the updated parameters will be used in the next round.
-	Get the Accuracy: Since with SGD, running a whole iteration, counting the correct divided by the number of whole data in dataset and we can get the accuracy.
