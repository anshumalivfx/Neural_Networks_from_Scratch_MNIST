# Neural Network from Scratch

This repository implements a simple neural network from scratch using NumPy. The network is trained on the MNIST dataset for digit recognition.

## Mathematics behind the Neural Network

### 1. **Data Loading**

The MNIST dataset is loaded from the '/kaggle/input/digit-recognizer/train.csv' file using Pandas.

### 2. **Data Preprocessing**

The data is shuffled and split into a development set (`data_dev`) and a training set (`data_train`). The pixel values are normalized to the range \([0, 1]\).

### 3. **Parameter Initialization**

The parameters of the neural network (weights and biases) are initialized randomly.

### 4. **Activation Functions**

#### Rectified Linear Unit (ReLU):

$$\text{ReLU}(Z) = \max(0, Z) \$$

#### Softmax:
$$ \text{softmax}(Z_i) = \frac{e^{Z_i}}{\sum_{j=1}^{10} e^{Z_j}} $$

### 5. **Forward Propagation**

The forward propagation step computes the activations at each layer:

$$ Z^{[1]} = W^{[1]} \cdot X + b^{[1]} $$

$$ A^{[1]} = \text{ReLU}(Z^{[1]}) $$

$$ Z^{[2]} = W^{[2]} \cdot A^{[1]} + b^{[2]} $$

$$ A^{[2]} = \text{softmax}(Z^{[2]}) $$

### 6. **Backward Propagation**

The backward propagation step computes the gradients with respect to the parameters:

$$ \frac{\partial \mathcal{L}}{\partial Z^{[2]}} = A^{[2]} - \text{one\_hot}(Y) $$

$$ \frac{\partial W^{[2]}}{\partial \mathcal{L}} = \frac{1}{m} \frac{\partial Z^{[2]}}{\partial \mathcal{L}} \cdot (A^{[1]})^T $$

$$ \frac{\partial b^{[2]}}{\partial \mathcal{L}} = \frac{1}{m} \sum_{i=1}^{m} \frac{\partial Z^{[2]}}{\partial \mathcal{L}} $$

$$ \frac{\partial Z^{[1]}}{\partial \mathcal{L}} = (W^{[2]})^T \cdot \frac{\partial Z^{[2]}}{\partial \mathcal{L}} \cdot \text{ReLU}'(Z^{[1]}) $$

$$ \frac{\partial W^{[1]}}{\partial \mathcal{L}} = \frac{1}{m} \frac{\partial Z^{[1]}}{\partial \mathcal{L}} \cdot X^T $$

$$ \frac{\partial b^{[1]}}{\partial \mathcal{L}} = \frac{1}{m} \sum_{i=1}^{m} \frac{\partial Z^{[1]}}{\partial \mathcal{L}} $$

### 7. **Parameter Update**

The parameters are updated using gradient descent:

$$ W^{[1]} = W^{[1]} - \alpha \frac{\partial W^{[1]}}{\partial \mathcal{L}} $$

$$ b^{[1]} = b^{[1]} - \alpha \frac{\partial b^{[1]}}{\partial \mathcal{L}} $$

$$ W^{[2]} = W^{[2]} - \alpha \frac{\partial W^{[2]}}{\partial \mathcal{L}} $$

$$ b^{[2]} = b^{[2]} - \alpha \frac{\partial b^{[2]}}{\partial \mathcal{L}} $$

### 8. **Training the Neural Network**

Gradient descent is used to train the neural network.

### 9. **Testing and Making Predictions**

The trained model is tested on a few examples, and predictions are made on the test dataset.

### 10. **Generating Submission File**

The final predictions on the test set are saved to a CSV file for submission.

## Dependencies

- NumPy
- pandas
- matplotlib
- tqdm

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.# Neural_Networks_from_Scratch_MNIST
