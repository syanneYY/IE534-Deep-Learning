# Implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch)

You should write your own code for convolutions (e.g., do not use SciPy's convolution function). The convolution network should have a single hidden layer with multiple channels. It should achieve at least 94% accuracy on the Test Set.

## Solutions
a.	The Test Accuracy is around 0.9521 on the test data as code shown, after training the model with 20 iterations on training data. All data are from MINIST dataset.


b.	The whole code of convolutional neural network with a single hidden layer of multiple channels follows the sequence of implementation : conv + activation(Relu) + fc + Softmax.



-	Processed with data: Load data -> normalization -> convert_to_onehot ( convert label Y )
-	Initialized Parameters initial_par () : X_input has C_in(input channel number) = 1, H_in(height of input X) = 28, W_in(width of input X) = 28; K has C_out(output channel number) = 8, K_h(height of K) = 7, K_w(width of K) = 7; W has n_output(labels of y) = 10, (H_out* W_out* C_out) = (22*22*8); b has n_output(labels of y) = 10. [X_in (28,28,1); Y(10); K(7,7,8); W(10,(22*22*8); b(10)]
-	Fed Data Froward: conv -> ReLU(activate it to non-linear) -> flatten -> fc -> Softmax; also record the output of each step in the cache.
-	Back Propagation: take gradient of Softmax -> fc -> flatten -> ReLU -> conv; also record gradients of each layer in grads
-	Update Parameters: update_par (update each parameter with different learning rate accounting to different iteration in each linear layer). Then the updated parameters will be used in the next round.
-	Get the Accuracy: Since with SGD, running a whole iteration, counting the correct divided by the number of whole data in dataset and we can get the accuracy.
