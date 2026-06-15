import numpy as np
import matplotlib.pyplot as plt


def init1(num_inputs, num_neurons):
    """Create the weight matrix for a single-layer neural network.

    The weights are initialized with small random values from the
    range [-0.1, 0.1) so that no neuron starts in a saturated region
    of the sigmoid and the neurons are not identical.

    Args:
        num_inputs: Number of inputs (features) to the network.
        num_neurons: Number of neurons (outputs / classes) in the layer.

    Returns:
        np.ndarray: A weight matrix of shape (num_inputs, num_neurons).
    """
    seed = 123456789                                  # fixed seed for reproducible weights
    np.random.seed(seed)                              # make the random draw deterministic
    matrix = np.random.rand(num_inputs, num_neurons)  # random values in the range [0, 1)
    scaled_matrix = matrix * 0.2 - 0.1                # rescale and shift them to [-0.1, 0.1)
    return scaled_matrix

num_inputs = 5                                        # example: number of input features
num_neurons = 3                                       # example: number of output neurons
weights = init1(num_inputs, num_neurons)              # build a sample weight matrix
print(weights)
print(f"Shape of weights matrix: {weights.shape}")
print(f"Min value: {weights.min():.4f}, Max value: {weights.max():.4f}")

def run1(weights, input_vector):
    """Run the forward pass of the single-layer network.

    Computes the weighted sum of the inputs and squashes it through
    the sigmoid activation function to produce the neuron outputs.

    Args:
        weights: Weight matrix of shape (num_inputs, num_neurons).
        input_vector: Input vector / matrix (one column per example).

    Returns:
        np.ndarray: The network outputs (activations) in the range (0, 1).
    """
    beta = 5.0                                        # slope of the sigmoid (steepness)

    if input_vector.ndim == 1 :                       # if a single example was passed as 1D
      input_vector = input_vector.reshape(-1,1)       # reshape it into a column vector
    net_input = weights.T @ input_vector              # weighted sum (net input) per neuron
    output = 1 / (1 + np.exp(-beta * net_input))      # sigmoid activation of the net input
    return output

def train1(weights_before, training_inputs, targets, num_iterations):
    """Train the single-layer network with the delta rule.

    For each iteration one random training example is selected, the
    network output is computed, the error is measured against the
    target, and the weights are nudged to reduce that error.

    Args:
        weights_before: Weight matrix before training.
        training_inputs: Training inputs, one example per column.
        targets: Target outputs, one example per column.
        num_iterations: Number of training iterations.

    Returns:
        np.ndarray: The weight matrix after training.
    """
    num_examples = training_inputs.shape[1]           # number of training examples (columns)
    num_outputs = targets.shape[0]                    # number of output neurons

    weights = weights_before.copy()                   # work on a copy, keep the original safe
    learning_rate = 0.1                               # step size of each weight update
    beta = 5.0                                         # sigmoid slope (matches run1)

    MSE_step = np.zeros(num_iterations)

    for step in range(num_iterations):                # repeat the update step many times
        example_idx = np.random.randint(num_examples)             # pick a random example
        input_vector = training_inputs[:,example_idx].reshape(-1,1)  # selected input as column
        output = run1(weights, input_vector)                      # forward pass -> output
        target = targets[:, example_idx].reshape(-1,1)            # desired output as column
        error = target - output                                   # output error (target - actual)
        delta = beta * error * output * (1-output)                # error times sigmoid derivative
        weight_update = learning_rate * (input_vector @ delta.T)  # weight correction (delta rule)
        weights = weights + weight_update                         # apply the correction
        MSE_step[step] = 0.5 * (1 / num_outputs) * (error.T @ error).item() 


    return weights , MSE_step

training_inputs = np.array([
    [4.0,    2.0,   -1.0  ],
    [0.01,  -1.0,    3.5  ],
    [0.01,   2.0,    0.01 ],
    [-1.0,   2.5,   -2.0  ],
    [-1.5,   2.0,    1.5  ]
])                                                    # training inputs: 5 features x 3 examples

targets = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])                                                    # one-hot targets: mammal / bird / fish

num_inputs = training_inputs.shape[0]                 # number of features (rows of training_inputs)
num_outputs = targets.shape[0]                        # number of classes (rows of targets)
weights_before = init1(num_inputs, num_outputs)       # initialize the weights
print("Initial weights:\n", weights_before)

output_before = run1(weights_before, training_inputs) # network outputs before any training
print("\nOutputs before training:\n", output_before)

num_epoch = 100                                       # how many training iterations to run
print(f"\n--- Training (train1 for num_epoch) ---")
weights_after, MSE = train1(weights_before, training_inputs, targets, num_epoch)  # train the network
print(f"Weights after training \n {weights_after}")

plt.figure(figsize=(10, 6))
plt.plot(MSE)
plt.title('MSE during training')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.grid(True)
plt.show()

output_after = run1(weights_after, training_inputs)   # network outputs after training

print("\nTarget outputs (targets):\n", targets)
np.set_printoptions(suppress=True)                    # print floats without scientific notation
print("\nOutputs after training (output_after):\n", output_after)
np.set_printoptions(suppress=False)                   # restore the default print options

me = ...

human = np.array([2, -5, -5, -5, -5])                 # feature vector for a new object: human
output_human = run1(weights_after, human)             # classify the human with trained weights

np.set_printoptions(suppress=True)
print("Result for human (Mammal / Bird / Fish):\n", output_human)

bat = np.array([4,-5,5,-5,-5])                        # feature vector for a new object: bat
output_bat = run1(weights_after, bat)                 # classify the bat with trained weights
np.set_printoptions(suppress=True)
print("output_bat\n", output_bat)
