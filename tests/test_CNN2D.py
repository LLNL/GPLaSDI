from lasdi.networks import CNN2D
from random import randint
import numpy as np
import torch

def test_set_data_shape():
    depth = 3
    layer_sizes = [[randint(2, 4), randint(45, 50), randint(45, 50)],
                   [randint(2, 4), randint(12, 15), randint(12, 15)],
                   [randint(2, 4), randint(2, 4), randint(2, 4)]]
    dilations = [1] * (depth-1)
    paddings = [randint(1, 3) for _ in range(depth-1)]
    strides = [randint(1, 3) for _ in range(depth-1)]

    input_shape = [randint(3, 5), randint(10, 20)] + layer_sizes[0]
    input_data = torch.rand(input_shape)

    cnnf = CNN2D(layer_sizes, 'forward',
                 strides, paddings, dilations)
    
    cnnf.set_data_shape(input_data.shape)
    cnnf.print_data_shape()

    assert(cnnf.batch_reshape[-3:] == input_shape[-3:])
    assert(np.prod(cnnf.batch_reshape) == np.prod(input_shape))
    
    output_data = cnnf(input_data)
    
    cnnb = CNN2D(layer_sizes[::-1], 'backward',
                 strides[::-1], paddings[::-1], dilations[::-1])
    cnnb.set_data_shape(input_data.shape)
    cnnb.print_data_shape()

    assert(cnnb.batch_reshape == input_shape)

    input_data0 = cnnb(output_data)
    assert(input_data0.shape == input_data.shape)

    return

def test_set_data_shape2():
    depth = 3
    layer_sizes = [[1, randint(45, 50), randint(45, 50)],
                   [randint(2, 4), randint(12, 15), randint(12, 15)],
                   [randint(2, 4), randint(2, 4), randint(2, 4)]]
    dilations = [1] * (depth-1)
    paddings = [randint(1, 3) for _ in range(depth-1)]
    strides = [randint(1, 3) for _ in range(depth-1)]

    def test_func(input_shape_):
        input_data = torch.rand(input_shape_)

        cnnf = CNN2D(layer_sizes, 'forward',
                     strides, paddings, dilations)
        
        cnnf.set_data_shape(input_data.shape)
        cnnf.print_data_shape()

        if ((len(input_shape_) == 2) or (input_shape_[-3] != 1)):
            assert(cnnf.batch_reshape[-2:] == input_shape_[-2:])
        else:
            assert(cnnf.batch_reshape[-3:] == input_shape_[-3:])
        
        assert(np.prod(cnnf.batch_reshape) == np.prod(input_shape_))
        
        output_data = cnnf(input_data)
        
        cnnb = CNN2D(layer_sizes[::-1], 'backward',
                     strides[::-1], paddings[::-1], dilations[::-1])
        cnnb.set_data_shape(input_data.shape)
        cnnb.print_data_shape()

        assert(cnnb.batch_reshape == input_shape_)

        input_data0 = cnnb(output_data)
        assert(input_data0.shape == input_data.shape)

    input_shape = layer_sizes[0][1:]
    test_func(input_shape)

    input_shape = [randint(3, 5), randint(10, 20)] + layer_sizes[0][1:]
    test_func(input_shape)

    input_shape = [randint(3, 5), randint(10, 20)] + layer_sizes[0]
    test_func(input_shape)

    return