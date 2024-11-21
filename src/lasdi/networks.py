import torch
import numpy as np

# activation dict
act_dict = {'ELU': torch.nn.ELU,
            'hardshrink': torch.nn.Hardshrink,
            'hardsigmoid': torch.nn.Hardsigmoid,
            'hardtanh': torch.nn.Hardtanh,
            'hardswish': torch.nn.Hardswish,
            'leakyReLU': torch.nn.LeakyReLU,
            'logsigmoid': torch.nn.LogSigmoid,
            'multihead': torch.nn.MultiheadAttention,
            'PReLU': torch.nn.PReLU,
            'ReLU': torch.nn.ReLU,
            'ReLU6': torch.nn.ReLU6,
            'RReLU': torch.nn.RReLU,
            'SELU': torch.nn.SELU,
            'CELU': torch.nn.CELU,
            'GELU': torch.nn.GELU,
            'sigmoid': torch.nn.Sigmoid,
            'SiLU': torch.nn.SiLU,
            'mish': torch.nn.Mish,
            'softplus': torch.nn.Softplus,
            'softshrink': torch.nn.Softshrink,
            'tanh': torch.nn.Tanh,
            'tanhshrink': torch.nn.Tanhshrink,
            'threshold': torch.nn.Threshold,
            }

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, layer_sizes,
                 act_type='sigmoid', reshape_index=None, reshape_shape=None,
                 threshold=0.1, value=0.0):
        super(MultiLayerPerceptron, self).__init__()

        # including input, hidden, output layers
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

        # Linear features between layers
        self.fcs = []
        for k in range(self.n_layers-1):
            self.fcs += [torch.nn.Linear(layer_sizes[k], layer_sizes[k + 1])]
        self.fcs = torch.nn.ModuleList(self.fcs)
        self.init_weight()

        # Reshape input or output layer
        assert((reshape_index is None) or (reshape_index in [0, -1]))
        assert((reshape_shape is None) or (np.prod(reshape_shape) == layer_sizes[reshape_index]))
        self.reshape_index = reshape_index
        self.reshape_shape = reshape_shape

        # Initalize activation function
        self.act_type = act_type
        if act_type == "threshold":
            self.act = act_dict[act_type](threshold, value)

        elif act_type == "multihead":
            raise RuntimeError("MultiLayerPerceptron: MultiheadAttention requires a different architecture!")

        #all other activation functions initialized here
        else:
            self.act = act_dict[act_type]()
        return
    
    def forward(self, x):
        if (self.reshape_index == 0):
            # make sure the input has a proper shape
            assert(list(x.shape[-len(self.reshape_shape):]) == self.reshape_shape)
            # we use torch.Tensor.view instead of torch.Tensor.reshape,
            # in order to avoid data copying.
            x = x.view(list(x.shape[:-len(self.reshape_shape)]) + [self.layer_sizes[self.reshape_index]])

        for i in range(self.n_layers-2):
            x = self.fcs[i](x) # apply linear layer
            x = self.act(x)

        x = self.fcs[-1](x)

        if (self.reshape_index == -1):
            # we use torch.Tensor.view instead of torch.Tensor.reshape,
            # in order to avoid data copying.
            x = x.view(list(x.shape[:-1]) + self.reshape_shape)

        return x
    
    def init_weight(self):
        # TODO(kevin): support other initializations?
        for fc in self.fcs:
            torch.nn.init.xavier_uniform_(fc.weight)
        return
    
    def print_architecture(self):
        print(self.layer_sizes)

class CNN2D(torch.nn.Module):
    from enum import Enum
    class Mode(Enum):
        Forward = 1
        Backward = -1

    def __init__(self, layer_sizes, mode,
                 strides, paddings, dilations,
                 groups=1, bias=True, padding_mode='zeros',
                 act_type='ReLU', data_shape=None):
        super(CNN2D, self).__init__()

        if (mode == 'forward'):
            self.mode = self.Mode.Forward
            module = torch.nn.Conv2d
        elif (mode == 'backward'):
            self.mode = self.Mode.Backward
            module = torch.nn.ConvTranspose2d
        else:
            raise RuntimeError('CNN2D: Unknown mode %s!' % mode)
        
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.channels = [layer_sizes[k][0] for k in range(self.n_layers)]

        # assert(len(kernel_sizes) == self.n_layers - 1)
        assert(len(strides) == self.n_layers - 1)
        assert(len(paddings) == self.n_layers - 1)
        assert(len(dilations) == self.n_layers - 1)
        # self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.dilations = dilations

        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

        from lasdi.networks import act_dict
        # TODO(kevin): not use threshold activation for now.
        assert(act_type != 'threshold')
        self.act = act_dict[act_type]()

        self.kernel_sizes = []
        self.fcs = []
        for k in range(self.n_layers - 1):
            kernel_size = self.compute_kernel_size(self.layer_sizes[k][1:], self.layer_sizes[k+1][1:],
                                                   self.strides[k], self.paddings[k], self.dilations[k], self.mode)
            out_shape = self.compute_output_layer_size(self.layer_sizes[k][1:], kernel_size, self.strides[k],
                                                       self.paddings[k], self.dilations[k], self.mode)
            assert(self.layer_sizes[k+1][1:] == out_shape)

            self.kernel_sizes += [kernel_size]
            self.fcs += [module(self.channels[k], self.channels[k+1], self.kernel_sizes[k],
                                stride=self.strides[k], padding=self.paddings[k], dilation=self.dilations[k],
                                groups=self.groups, bias=self.bias, padding_mode=self.padding_mode)]
            
        self.fcs = torch.nn.ModuleList(self.fcs)
        self.init_weight()

        if (data_shape is not None):
            self.set_data_shape(data_shape)

        return
    
    def set_data_shape(self, data_shape : list):
        idx = 0 if (self.mode == CNN2D.Mode.Forward) else -1

        if (self.mode == CNN2D.Mode.Forward):
            if (len(data_shape) > 2):
                if (data_shape[-3] != self.channels[idx]):
                    assert(self.channels[idx] == 1)
                    self.batch_reshape = [np.prod(data_shape[:-2]), 1]
                else:
                    self.batch_reshape = [np.prod(data_shape[:-3]), self.channels[idx]]
            elif (len(data_shape) == 2):
                assert(self.channels[idx] == 1)
                self.batch_reshape = [1]

            self.batch_reshape += data_shape[-2:]
        else:
            self.batch_reshape = list(data_shape)
        return
    
    def print_data_shape(self):
        mode_str = "forward" if (self.mode == CNN2D.Mode.Forward) else "backward"
        print("mode: ", mode_str)
        print("batch reshape: ", self.batch_reshape)
        for k in range(self.n_layers - 1):
            print('input layer: ', self.layer_sizes[k],
                  'kernel size: ', self.kernel_sizes[k],
                  'output layer: ', self.layer_sizes[k+1])
        return
    
    def forward(self, x):
        if ((self.batch_reshape is not None) and (self.mode == CNN2D.Mode.Forward)):
            x = x.view(self.batch_reshape)

        for i in range(self.n_layers-2):
            x = self.fcs[i](x)
            x = self.act(x)

        x = self.fcs[-1](x)

        if ((self.batch_reshape is not None) and (self.mode == CNN2D.Mode.Backward)):
            x = x.view(self.batch_reshape)
        return x
    
    @classmethod
    def compute_kernel_size(cls, input_shape, output_shape, stride, padding, dilation, mode):
        assert(len(input_shape) == 2)
        assert(len(output_shape) == 2)
        if (type(stride) is int):
            stride = [stride, stride]
        if (type(padding) is int):
            padding = [padding, padding]
        if (type(dilation) is int):
            dilation = [dilation, dilation]

        if (mode == CNN2D.Mode.Forward):
            kern_H = (input_shape[0] + 2 * padding[0] - 1 - stride[0] * (output_shape[0] - 1)) / dilation[0] + 1
            kern_W = (input_shape[1] + 2 * padding[1] - 1 - stride[1] * (output_shape[1] - 1)) / dilation[1] + 1
        elif (mode == CNN2D.Mode.Backward):
            kern_H = (output_shape[0] - (input_shape[0] - 1) * stride[0] + 2 * padding[0] - 1) / dilation[0] + 1
            kern_W =  (output_shape[1] - (input_shape[1] - 1) * stride[1] + 2 * padding[1] - 1) / dilation[1] + 1
        else:
            raise RuntimeError('CNN2D: Unknown mode %s!' % mode)
        
        if ((kern_H <= 0) or (kern_W <= 0)):
            print("input shape: ", input_shape)
            print("output shape: ", output_shape)
            print("stride: ", stride)
            print("padding: ", padding)
            print("dilation: ", dilation)
            print("resulting kernel size: ", [int(np.floor(kern_H)), int(np.floor(kern_W))])
            raise RuntimeError("CNN2D.compute_kernel_size: no feasible kernel size. "
                               "Adjust the architecture!")
        
        return [int(np.floor(kern_H)), int(np.floor(kern_W))]
    
    @classmethod
    def compute_input_layer_size(cls, output_shape, kernel_size, stride, padding, dilation, mode):
        assert(len(output_shape) == 2)
        if (type(kernel_size) is int):
            kernel_size = [kernel_size, kernel_size]
        if (type(stride) is int):
            stride = [stride, stride]
        if (type(padding) is int):
            padding = [padding, padding]
        if (type(dilation) is int):
            dilation = [dilation, dilation]

        if (mode == cls.Mode.Forward):
            Hin = (output_shape[0] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + 1
            Win = (output_shape[1] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + 1
        elif (mode == cls.Mode.Backward):
            Hin = (output_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
            Win = (output_shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
        else:
            raise RuntimeError('CNN2D: Unknown mode %s!' % mode)
        
        return [int(np.floor(Hin)), int(np.floor(Win))]

    @classmethod
    def compute_output_layer_size(cls, input_shape, kernel_size, stride, padding, dilation, mode):
        assert(len(input_shape) == 2)
        if (type(kernel_size) is int):
            kernel_size = [kernel_size, kernel_size]
        if (type(stride) is int):
            stride = [stride, stride]
        if (type(padding) is int):
            padding = [padding, padding]
        if (type(dilation) is int):
            dilation = [dilation, dilation]

        if (mode == cls.Mode.Forward):
            Hout = (input_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
            Wout = (input_shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
        elif (mode == cls.Mode.Backward):
            Hout = (input_shape[0] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + 1
            Wout = (input_shape[1] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + 1
        else:
            raise RuntimeError('CNN2D: Unknown mode %s!' % mode)
        
        if ((mode == cls.Mode.Forward) and ((Hout > np.floor(Hout)) or (Wout > np.floor(Wout)))):
            print("input shape: ", input_shape)
            print("kernel size: ", kernel_size)
            print("stride: ", stride)
            print("padding: ", padding)
            print("dilation: ", dilation)
            print("resulting output shape: ", [Hout, Wout])
            raise RuntimeError("CNN2D.compute_output_layer_size: given architecture will not return the same size backward. "
                               "Adjust the architecture!")

        return [int(np.floor(Hout)), int(np.floor(Wout))]
    
    def init_weight(self):
        # TODO(kevin): support other initializations?
        for fc in self.fcs:
            torch.nn.init.xavier_uniform_(fc.weight)
        return