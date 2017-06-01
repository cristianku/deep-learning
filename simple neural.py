import numpy as np

input_data = np.array([-1,2])
weights = {
            'node_0':np.array([3,3]),
            'node_1': np.array([3, 3]),
            'output': np.array([3, 3])

}



def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(0, input)

    # Return the value just calculated
    return(output)

# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

print 'node_0_output'
print node_0_output

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

print 'node_1_output'
print node_1_output

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

print 'hidden_layer_outputs'
print hidden_layer_outputs

# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()

# Print model output
print(model_output)
