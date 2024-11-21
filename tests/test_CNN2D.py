from lasdi.networks import CNN2D
from random import randint
import numpy as np
import torch

def test_compute_output_layer():
    depth = 2
    channels = [randint(2, 4) for _ in range(depth)]
    strides = [randint(1, 3) for _ in range(depth-1)]
    paddings = [randint(1, 3) for _ in range(depth-1)]
    dilations = [randint(1, 3) for _ in range(depth-1)]
    kernel_sizes = [[randint(4, 6) for _ in range(2)]]

    output_shape = [randint(10, 20), randint(10, 20)]
    output_data = torch.rand([channels[-1]] + output_shape)

    cnnb = CNN2D('backward', channels[::-1], kernel_sizes[::-1],
                strides[::-1], paddings[::-1], dilations[::-1])
    
    input_shape = CNN2D.compute_output_layer_size(output_data.shape[1:], kernel_sizes[0],
                                                  strides[0], paddings[0], dilations[0], cnnb.mode)
    input_data = cnnb(output_data)
    assert(list(input_data.shape[-2:]) == input_shape)

    cnnf = CNN2D('forward', channels, kernel_sizes,
                strides, paddings, dilations)

    output_shape0 = CNN2D.compute_output_layer_size(input_data.shape[1:], kernel_sizes[0],
                                                   strides[0], paddings[0], dilations[0], cnnf.mode)
    assert(output_shape0 == output_shape)
    
    output_data0 = cnnf(input_data)
    assert(output_data.shape == output_data0.shape)
    
    return

def test_compute_input_layer():
    depth = 2
    channels = [randint(2, 4) for _ in range(depth)]
    strides = [randint(1, 3) for _ in range(depth-1)]
    paddings = [randint(1, 3) for _ in range(depth-1)]
    dilations = [randint(1, 3) for _ in range(depth-1)]
    kernel_sizes = [[randint(4, 6) for _ in range(2)]]

    output_shape = [randint(10, 20), randint(10, 20)]

    cnnf = CNN2D('forward', channels, kernel_sizes,
                strides, paddings, dilations)
    
    input_shape = CNN2D.compute_input_layer_size(output_shape, kernel_sizes[0],
                                                   strides[0], paddings[0], dilations[0], cnnf.mode)
    input_data = torch.rand([channels[0]] + input_shape)

    output_data = cnnf(input_data)
    assert(list(output_data.shape[-2:]) == output_shape)

    cnnb = CNN2D('backward', channels[::-1], kernel_sizes[::-1],
                strides[::-1], paddings[::-1], dilations[::-1])
    
    output_shape0 = CNN2D.compute_input_layer_size(input_data.shape[1:], kernel_sizes[0],
                                                   strides[0], paddings[0], dilations[0], cnnb.mode)
    assert(output_shape0 == output_shape)

    input_data0 = cnnb(output_data)
    assert(input_data.shape == input_data0.shape)
    return

def test_set_data_shape():
    depth = 3
    channels = [randint(2, 4) for _ in range(depth)]
    strides = [1] * (depth-1)
    paddings = [randint(1, 3) for _ in range(depth-1)]
    dilations = [randint(1, 3) for _ in range(depth-1)]
    kernel_sizes = [[randint(4, 6), randint(4, 6)]] + [randint(2, 3) for _ in range(depth-2)]

    input_shape = [randint(3, 5), randint(10, 20), channels[0], randint(50, 60), randint(50, 60)]
    input_data = torch.rand(input_shape)

    cnnf = CNN2D('forward', channels, kernel_sizes,
                 strides, paddings, dilations)
    
    cnnf.set_data_shape(input_data.shape)
    cnnf.print_data_shape()

    assert(cnnf.batch_reshape[-3:] == input_shape[-3:])
    assert(np.prod(cnnf.batch_reshape) == np.prod(input_shape))
    
    output_data = cnnf(input_data)
    
    cnnb = CNN2D('backward', channels[::-1], kernel_sizes[::-1],
                 strides[::-1], paddings[::-1], dilations[::-1])
    cnnb.set_data_shape(input_data.shape)
    cnnb.print_data_shape()

    assert(cnnb.batch_reshape == input_shape)

    input_data0 = cnnb(output_data)
    assert(input_data0.shape == input_data.shape)

    return

def test_set_data_shape2():
    depth = 3
    channels = [1] + [randint(2, 4) for _ in range(depth-1)]
    strides = [1] * (depth-1)
    paddings = [randint(1, 3) for _ in range(depth-1)]
    dilations = [randint(1, 3) for _ in range(depth-1)]
    kernel_sizes = [[randint(4, 6), randint(4, 6)]] + [randint(2, 3) for _ in range(depth-2)]

    def test_func(input_shape_):
        input_data = torch.rand(input_shape_)

        cnnf = CNN2D('forward', channels, kernel_sizes,
                    strides, paddings, dilations)
        
        cnnf.set_data_shape(input_data.shape)
        cnnf.print_data_shape()

        if ((len(input_shape_) == 2) or (input_shape_[-3] != 1)):
            assert(cnnf.batch_reshape[-2:] == input_shape_[-2:])
        else:
            assert(cnnf.batch_reshape[-3:] == input_shape_[-3:])
        
        assert(np.prod(cnnf.batch_reshape) == np.prod(input_shape_))
        
        output_data = cnnf(input_data)
        
        cnnb = CNN2D('backward', channels[::-1], kernel_sizes[::-1],
                    strides[::-1], paddings[::-1], dilations[::-1])
        cnnb.set_data_shape(input_data.shape)
        cnnb.print_data_shape()

        assert(cnnb.batch_reshape == input_shape_)

        input_data0 = cnnb(output_data)
        assert(input_data0.shape == input_data.shape)

    input_shape = [randint(50, 60), randint(50, 60)]
    test_func(input_shape)

    input_shape = [randint(3, 5), randint(10, 20), randint(50, 60), randint(50, 60)]
    test_func(input_shape)

    input_shape = [randint(3, 5), randint(10, 20), channels[0], randint(50, 60), randint(50, 60)]
    test_func(input_shape)

    return