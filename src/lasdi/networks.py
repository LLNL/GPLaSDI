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