import pickle
import os


def relu(x):
    return max(0, x)

# Forward Pass Functions
def forward_pass_hidden(W1, b1, inputs):
    hidden_outputs = [relu(sum(inputs[i] * W1[i][j] for i in range(len(inputs))) + b1[j]) for j in range(len(W1[0]))]
    return hidden_outputs

def forward_pass_output(W2, b2, hidden_outputs):
    return relu(sum(hidden_outputs[i] * W2[i] for i in range(len(hidden_outputs))) + b2)

# Load the trained model from a file
def load_weights_biases(filename):
    current_directory = os.path.dirname(__file__)  
    file_path = os.path.join(current_directory, 'uploadfiles', filename)

    # Load the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    return data['W1'], data['b1'], data['W2'], data['b2']

# Prediction function using the loaded model
def predict(inputs, W1, b1, W2, b2):
    hidden_outputs = forward_pass_hidden(W1, b1, inputs)
    output = forward_pass_output(W2, b2, hidden_outputs)
    print(output)
    return output


def processGrid(grid):
    # print(grid)
    row_sums = [sum(map(int, row)) for row in grid]
    col_sums = [sum(int(grid[row][col]) for row in range(7)) for col in range(5)]
    results = row_sums + col_sums
    print(f"Concat Value: {results}")

    return row_sums + col_sums


def inputGrid(grid):
    gridResult = processGrid(grid)
    W1, b1, W2, b2 = load_weights_biases('currentBiases.pkl')
    hidden_outputs = forward_pass_hidden(W1, b1, gridResult)
    output = forward_pass_output(W2, b2, hidden_outputs)
    print(output)
    return output