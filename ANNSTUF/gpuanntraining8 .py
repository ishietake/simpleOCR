import cupy as cp
import pandas as pd

def initialize_values(num_inputs, num_hidden):
    min_val = -(cp.sqrt(6) / cp.sqrt(num_inputs + num_hidden))
    max_val = (cp.sqrt(6) / cp.sqrt(num_inputs + num_hidden))
    W1 = cp.random.uniform(min_val, max_val, (num_inputs, num_hidden))
    b1 = cp.random.uniform(min_val, max_val, num_hidden)
    W2 = cp.random.uniform(min_val, max_val, num_hidden)
    b2 = cp.random.uniform(min_val, max_val)
    return W1, b1, W2, b2

def relu(x):
    return cp.maximum(0, x)

def relu_derivative(x):
    return cp.where(x > 0, 1, 0)

def forward_pass_hidden(W1, b1, inputs):
    net_input = cp.dot(inputs, W1) + b1
    return relu(net_input)

def forward_pass_output(W2, b2, hidden_outputs):
    net_input = cp.dot(hidden_outputs, W2) + b2
    return relu(net_input)

def dError(output, target):
    return 2 * (output - target)

def values(output, target, input):
    return dError(output, target) * relu_derivative(output) * input

def hiddenvalues(output, target, weight, output1, input):
    return dError(output, target) * relu_derivative(output) * weight * relu_derivative(output1) * input

def update_hidden_weights(output, target, hidden_outputs, inputs, W1, b1, W2, rate):
    for j in range(W1.shape[1]):
        for i in range(inputs.shape[0]):
            W1[i, j] -= rate * hiddenvalues(output, target, W2[j], hidden_outputs[j], inputs[i])
        b1[j] -= rate * hiddenvalues(output, target, W2[j], hidden_outputs[j], 1)
    return W1, b1

def update_output_weights(output, target, hidden_outputs, W2, b2, rate):
    for i in range(hidden_outputs.shape[0]):
        W2[i] -= rate * values(output, target, hidden_outputs[i])
        b2 -= rate * values(output, target, 1)
    return W2, b2

def preprocess_grid(grid):
    row_sums = cp.array([cp.sum(cp.array(row, dtype=int)) for row in grid])
    col_sums = cp.array([cp.sum(cp.array([int(grid[row][col]) for row in range(7)])) for col in range(5)])
    print(f"Concat Value: {cp.concatenate((row_sums, col_sums))}")
    return cp.concatenate((row_sums, col_sums))

def read_training_data(file_path):
    df = pd.read_excel(file_path, header=None)
    training_data = []

    num_rows = df.shape[0]
    for i in range(0, num_rows, 8): 
        grid = df.iloc[i:i+7, 0:5].values.tolist()
        target = df.iloc[i+7, 5]
        inputs = preprocess_grid(grid)
        training_data.append((inputs, target))
        print(f"Grid: {grid}")
        print(f"Processed Inputs: {inputs}")
        print(f"Target: {target}")
    return training_data

def train(file_path, epochs, desired_error, rate=0.000001):
    training_data = read_training_data(file_path)
    num_inputs = 12  
    num_hidden = 50
    
    W1, b1, W2, b2 = initialize_values(num_inputs, num_hidden)
    epoch = 0
    total_error = float('inf')

    while epoch < epochs and total_error > desired_error:
        total_error = 0
        for inputs, target in training_data:
            hidden_outputs = forward_pass_hidden(W1, b1, inputs)
            output = forward_pass_output(W2, b2, hidden_outputs)
            W2, b2 = update_output_weights(output, target, hidden_outputs, W2, b2, rate)
            W1, b1 = update_hidden_weights(output, target, hidden_outputs, inputs, W1, b1, W2, rate)

            error = 0.5 * (output - target) ** 2
            total_error += error

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Error: {total_error}")

        epoch += 1

    print(f"Final weights: W1 = {W1}, b1 = {b1}, W2 = {W2}, b2 = {b2}")

    for inputs, target in training_data:
        hidden_outputs = forward_pass_hidden(W1, b1, inputs)
        output = forward_pass_output(W2, b2, hidden_outputs)
        print(f"Input: {inputs}, Predicted Output: {output}, Target: {target}")

if __name__ == "__main__":
    file_path = 'datasetForANN.xlsx'  
    epochs = 100000
    desired_error = 0.0000000000001
    train(file_path, epochs, desired_error)
