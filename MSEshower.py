import matplotlib.pyplot as plt
import numpy as np


def plot_two_arrays(MSEtrain, MSEtest):
    # Check if the input arrays are of the same length
    if len(MSEtrain) != len(MSEtest):
        raise ValueError("Input arrays must be of the same length")

    # Generate the index for the x-axis
    index = np.arange(len(MSEtrain))

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(index, MSEtrain, label='MSE_train', marker='o')
    plt.plot(index, MSEtest, label='MSE_test', marker='s')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Line Plot of Two Arrays')

    # Add legend
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
# Example usage
    array1 = [1, 3, 5, 7, 9]
    array2 = [2, 4, 6, 8, 10]
    plot_two_arrays(array1, array2)