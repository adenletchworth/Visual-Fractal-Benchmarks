# Import necessary modules and definitions for complex arithmetic and utilities.
import benchmark
from complex import ComplexSIMD, ComplexFloat64
from math import iota, abs
from python import Python
from sys.info import num_physical_cores
from algorithm import parallelize, vectorize
from tensor import Tensor
from utils.index import Index

alias MAX_ITERS = 200
alias width = 960
alias height = 960
alias float_type = DType.float64
alias min_x = -2.0
alias max_x = 1.2
alias min_y = -1.8
alias max_y = 1.2

def burningship_kernel(c: ComplexFloat64) -> Int:
    z = c
    for i in range(MAX_ITERS):
        z = ComplexFloat64(abs(z.re), abs(z.im)) * ComplexFloat64(abs(z.re), abs(z.im)) + c
        if z.squared_norm() > 4:
            return i  
    return MAX_ITERS  

def compute_burningship() -> Tensor[float_type]:
    # create a matrix. Each element of the matrix corresponds to a pixel
    t = Tensor[float_type](height, width)

    dx = (max_x - min_x) / width
    dy = (max_y - min_y) / height

    y = min_y
    for row in range(height):
        x = min_x
        for col in range(width):
            t[Index(row, col)] = burningship_kernel(ComplexFloat64(x, y))
            x += dx
        y += dy
    return t

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

    image = light.shade(numpy_arr, plt.cm.autumn, colors.PowerNorm(0.3), 'hsv', 0, 0, 1.5)
    plt.imshow(image)
    plt.axis("off")  
    plt.savefig('./Images/burningshipfractal.png')  
    plt.show()  

def main() -> None:
    t = compute_burningship()
    show_plot(t)  







