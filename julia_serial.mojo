# Import necessary modules and definitions for complex arithmetic and utilities.
import benchmark
from complex import ComplexSIMD, ComplexFloat64
from math import iota
from python import Python
from sys.info import num_physical_cores
from algorithm import parallelize, vectorize
from tensor import Tensor
from utils.index import Index

# Define constants used in the computation of the Julia set.
alias MAX_ITERS = 200
alias width = 960
alias height = 960
alias float_type = DType.float64
alias min_x = -1.7
alias max_x = 1.7
alias min_y = -1.5
alias max_y = 1.5

# Function to compute the Julia set for a given complex number z with parameter c.
def julia_kernel(z: ComplexFloat64, c: ComplexFloat64) -> Int:
    for i in range(MAX_ITERS):
        z = z * z + c  
        if z.squared_norm() > 4:
            return i  # Return the iteration number if z escapes.
    return MAX_ITERS  # Return the max iterations if z does not escape.

# Function to compute the Julia set for each point on a 2D grid.
def compute_julia() -> Tensor[float_type]:
    t = Tensor[float_type](height, width)  # Initialize tensor for storing results.
    dx = (max_x - min_x) / width  # Calculate horizontal step size.
    dy = (max_y - min_y) / height  # Calculate vertical step size.
    
    # Iterate over each point in the grid, compute the corresponding complex number.
    for i in range(height):
        for j in range(width):
            x = min_x + j * dx
            y = min_y + i * dy
            z = ComplexFloat64(x, y)
            # Store the escape time for each point in the tensor.
            t[Index(i, j)] = julia_kernel(z, ComplexFloat64(-0.8, 0.156))
    return t

# Function to visualize Julia Set
def show_plot(tensor: Tensor[float_type]) -> None:
    alias scale = 10
    alias dpi = 64

    # Import python libraries
    np = Python.import_module('numpy')
    plt = Python.import_module('matplotlib.pyplot')
    colors = Python.import_module('matplotlib.colors')

    # Convert tensor data to a numpy array
    numpy_arr = np.zeros((height, width), np.float64)
    for row in range(height):
        for col in range(width):
            numpy_arr.itemset((col, row), tensor[col, row])

    fig = plt.figure(1, [scale, scale * height // width], dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], False, 1)
    light = colors.LightSource(315, 10, 0, 1, 1, 0)

    image = light.shade(numpy_arr, plt.cm.magma, colors.PowerNorm(0.3), 'hsv', 0, 0, 1.5)
    plt.imshow(image)
    plt.axis("off")  
    plt.savefig('./Images/juliafractal.png')  
    plt.show()  

# Main function to execute the computation and visualization of the Julia set.
def main() -> None:
    t = compute_julia()
    show_plot(t)  # Call to display the computed Julia set.







