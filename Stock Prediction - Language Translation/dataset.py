'''
E4040 2024Fall Assignment3

'''
import numpy as np

def create_xor_dataset(num_samples, seq_len=8):
    """
    This function creates a dataset needed for XOR network. It outputs the input(data) and the corresponding tag(output)
    of the XOR network.

    :param num_samples: The total number of samples you would like for training.
    :param seq_len: The length of each training input. This determines the second dimension of data.

    :return data: A randomly generated numpy matrix with size [num_samples, seq_len] that only contains 0 and 1.
    :return output: A numpy matrix with size [num_samples, seq_len]. The value of this matrix follows:
                    output[i][j] = data[i][0] XOR data[i][1] XOR data[i][2] XOR ... XOR data[i][j]

    """

    data = np.random.randint(2, size=(num_samples, seq_len, 1))
    output = np.zeros([num_samples, seq_len], dtype=int)

    for i in range(seq_len):
        # for xor
        output[:, i] = np.mod(np.sum(data[:, :i+1, 0], axis=1), 2)
        # for xnor
        #output[:, i] = 1 - np.mod(np.sum(data[:, :i+1, 0], axis=1), 2)

    return data, output
