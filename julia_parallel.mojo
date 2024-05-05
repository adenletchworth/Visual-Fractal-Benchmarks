import benchmark
from complex import ComplexSIMD, ComplexFloat64
from math import iota
from python import Python
from sys.info import num_physical_cores
from algorithm import parallelize, vectorize
from tensor import Tensor
from utils.index import Index

alias float_type = DType.float64
alias simd_width = 2 * simdwidthof[float_type]()

alias MAX_ITERS = 200
alias width = 960
alias height = 960
alias min_x = -1.7
alias max_x = 1.7
alias min_y = -1.5
alias max_y = 1.5

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

    image = light.shade(numpy_arr, plt.cm.plasma, colors.PowerNorm(0.3), 'hsv', 0, 0, 1.5)
    plt.imshow(image)
    plt.axis("off")  
    plt.savefig('./Images/juliafractal1.png')  
    plt.show()  

fn julia_kernel_SIMD[
    simd_width: Int
](cx: SIMD[float_type, simd_width], cy: SIMD[float_type, simd_width]) -> SIMD[float_type, simd_width]:
    """A vectorized implementation of the inner Julia set computation."""
    var x = cx
    var y = cy
    var y2 = SIMD[float_type, simd_width](0)
    var iters = SIMD[float_type, simd_width](0)
    var c_re = .37
    var c_im = 0.1

    var t: SIMD[DType.bool, simd_width] = True
    for i in range(MAX_ITERS):
        if not t.reduce_or():
            break
        y2 = y*y
        y = x.fma(y + y, c_im)
        t = x.fma(x, y2) <= 4
        x = x.fma(x, c_re - y2)
        iters = t.select(iters + 1, iters)
    return iters

fn parallelized():
    var t = Tensor[float_type](height, width)

    @parameter
    fn worker(row: Int):
        var scale_x = (max_x - min_x) / width
        var scale_y = (max_y - min_y) / height

        @__copy_capture(scale_x, scale_y)
        @parameter
        fn compute_vector[simd_width: Int](col: Int):
            """Each time we operate on a `simd_width` vector of pixels."""
            var cx = min_x + (col + iota[float_type, simd_width]()) * scale_x
            var cy = min_y + row * scale_y
            var c = ComplexSIMD[float_type, simd_width](cx, cy)
            t.data().store(row * width + col, julia_kernel_SIMD[simd_width](cx, cy))

        # Vectorize the call to compute_vector where call gets a chunk of pixels.
        vectorize[compute_vector, simd_width](width)


    @parameter
    fn bench_parallel[simd_width: Int]():
        parallelize[worker](height, height)

    var parallelized = benchmark.run[bench_parallel[simd_width]](
        max_runtime_secs=0.5
    ).mean(benchmark.Unit.ms)

    print("Parallelized:", parallelized, benchmark.Unit.ms)

    try:
        _ = show_plot(t)
    except e:
        print("failed to show plot:", e)

def main() -> None:
    parallelized()

