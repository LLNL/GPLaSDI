import torch
import numpy as np

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
""":obj:`dict` : Dictionary to activation functions.

- :obj:`'ELU'`: :obj:`torch.nn.ELU`
- :obj:`'hardshrink'`: :obj:`torch.nn.Hardshrink`
- :obj:`'hardsigmoid'`: :obj:`torch.nn.Hardsigmoid`
- :obj:`'hardtanh'`: :obj:`torch.nn.Hardtanh`
- :obj:`'hardswish'`: :obj:`torch.nn.Hardswish`
- :obj:`'leakyReLU'`: :obj:`torch.nn.LeakyReLU`
- :obj:`'logsigmoid'`: :obj:`torch.nn.LogSigmoid`
- :obj:`'multihead'`: :obj:`torch.nn.MultiheadAttention`
- :obj:`'PReLU'`: :obj:`torch.nn.PReLU`
- :obj:`'ReLU'`: :obj:`torch.nn.ReLU`
- :obj:`'ReLU6'`: :obj:`torch.nn.ReLU6`
- :obj:`'RReLU'`: :obj:`torch.nn.RReLU`
- :obj:`'SELU'`: :obj:`torch.nn.SELU`
- :obj:`'CELU'`: :obj:`torch.nn.CELU`
- :obj:`'GELU'`: :obj:`torch.nn.GELU`
- :obj:`'sigmoid'`: :obj:`torch.nn.Sigmoid`
- :obj:`'SiLU'`: :obj:`torch.nn.SiLU`
- :obj:`'mish'`: :obj:`torch.nn.Mish`
- :obj:`'softplus'`: :obj:`torch.nn.Softplus`
- :obj:`'softshrink'`: :obj:`torch.nn.Softshrink`
- :obj:`'tanh'`: :obj:`torch.nn.Tanh`
- :obj:`'tanhshrink'`: :obj:`torch.nn.Tanhshrink`
- :obj:`'threshold'`: :obj:`torch.nn.Threshold`
"""

class MultiLayerPerceptron(torch.nn.Module):
    """Vanilla multi-layer perceptron neural networks module.
    """

    def __init__(self, layer_sizes,
                 act_type='sigmoid', reshape_index=None, reshape_shape=None,
                 threshold=0.1, value=0.0):
        """
        Parameters
        ----------
        layer_sizes : :obj:`list(int)`
            List of vector dimensions of layers.
        act_type : :obj:`str`, optional
            Type of activation functions. By default :obj:`'sigmoid'` is used.
            See :obj:`act_dict` for available types.
        reshape_index : :obj:`int`, optinal
            Index of layer to reshape input/output data. Either 0 or -1 is allowed.

            - 0 : the first (input) layer
            - -1 : the last (output) layer

            By default the index is :obj:`None`, and reshaping is not executed.
        reshape_shape : :obj:`list(int)`, optional
            Target shape from/to which input/output data is reshaped.
            Reshaping behavior changes by :attr:`reshape_index`.
            By default the index is :obj:`None`, and reshaping is not executed.
            For details on reshaping action, see :attr:`reshape_shape`.

        Note
        ----
        :obj:`numpy.prod(reshape_shape) == layer_sizes[reshape_index]`

        """
        super(MultiLayerPerceptron, self).__init__()
        
        self.n_layers = len(layer_sizes)
        """:obj:`int` : Depth of layers including input, hidden, output layers."""
        
        self.layer_sizes = layer_sizes
        """:obj:`list(int)` : Vector dimensions corresponding to each layer."""

        self.fcs = []
        """:obj:`torch.nn.ModuleList` : linear features between layers."""
        for k in range(self.n_layers-1):
            self.fcs += [torch.nn.Linear(layer_sizes[k], layer_sizes[k + 1])]
        self.fcs = torch.nn.ModuleList(self.fcs)

        self.init_weight()

        # Reshape input or output layer
        assert(reshape_index in [0, -1, None])
        assert((reshape_shape is None) or (np.prod(reshape_shape) == layer_sizes[reshape_index]))
        self.reshape_index = reshape_index
        """:obj:`int` : Index of layer to reshape input/output data.

        - 0 : the first (input) layer
        - -1 : the last (output) layer
        - :obj:`None` : no reshaping
        """
        self.reshape_shape = reshape_shape
        """:obj:`list(int)` : Target shape from/to which input/output data is reshaped.
        For a reshape_shape :math:`[R_1, R_2, \ldots, R_n]`,

        - if :attr:`reshape_index` is 0, the input data shape is changed as

        .. math::
            [\ldots, R_1, R_2, \ldots, R_n] \\longrightarrow [\ldots, \prod_{i=1}^n R_i]

        - if :attr:`reshape_index` is -1, the output data shape is changed as

        .. math::
            [\ldots, \prod_{i=1}^n R_i] \\longrightarrow [\ldots, R_1, R_2, \ldots, R_n]

        - :obj:`None` : no reshaping
        """

        # Initalize activation function
        self.act_type = act_type
        """:obj:`str` : Type of activation functions."""

        if act_type == "threshold":
            act = act_dict[act_type](threshold, value)
            """:obj:`torch.nn.Module` : Activation function used throughout the layers."""

        elif act_type == "multihead":
            raise RuntimeError("MultiLayerPerceptron: MultiheadAttention requires a different architecture!")

        #all other activation functions initialized here
        else:
            act = act_dict[act_type]()

        self.act = act
        """:obj:`torch.nn.Module` : Activation function used throughout the layers."""
        return
    
    def forward(self, x):
        """Evaluate through the module.
        
        Parameters
        ----------
        x : :obj:`torch.Tensor`
            Input data to pass into the module.

        Note
        ----
        For :attr:`reshape_index` =0,
        the last :math:`n` dimensions of :obj:`x` must match
        :attr:`reshape_shape` :math:`=[R_1, R_2, \ldots, R_n]`.

        Returns
        -------
        :obj:`torch.Tensor`
            Output tensor evaluated from the module.

        Note
        ----
        For :attr:`reshape_index` =-1,
        the last dimension of the output tensor will be reshaped as
        :attr:`reshape_shape` :math:`=[R_1, R_2, \ldots, R_n]`.
        
        """
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
        """Initialize weights of linear features according to Xavier uniform distribution."""

        # TODO(kevin): support other initializations?
        for fc in self.fcs:
            torch.nn.init.xavier_uniform_(fc.weight)
        return
    
    def print_architecture(self):
        """Print out the architecture of the module."""
        print(self.layer_sizes)

class CNN2D(torch.nn.Module):
    """Two-dimensional convolutional neural networks."""

    from enum import Enum
    class Mode(Enum):
        """Enumeration to specify direction of CNN."""
        Forward = 1
        """Contracting direction"""
        Backward = -1
        """Expanding direction"""

    def __init__(self, layer_sizes, mode,
                 strides, paddings, dilations,
                 groups=1, bias=True, padding_mode='zeros',
                 act_type='ReLU', data_shape=None):
        """
        Parameters
        ----------
        layer_sizes : :obj:`numpy.array`
            2d array of tensor dimension of each layer.
            See :attr:`layer_sizes`.
        mode : :obj:`str`
            Direction of CNN
            - `forward`: contracting direction
            - `backward`: expanding direction
        strides : :obj:`list`
            List of strides corresponding to each layer.
            Each stride is either integer or tuple.
        paddings : :obj:`list`
            List of paddings corresponding to each layer.
            Each padding is either integer or tuple.
        dilations : :obj:`list`
            List of dilations corresponding to each layer.
            Each dilation is either integer or tuple.
        groups : :obj:`int`, optional
            Groups that applies to all layers. By default 1
        bias : :obj:`bool`, optional
            Bias that applies to all layers. By default :obj:`True`
        padding_mode : :obj:`str`, optional
            Padding_mode that applies to all layers. By default :obj:`'zeros'`
        act_type : :obj:`str`, optional
            Activation function applied between all layers. By default :obj:`'ReLU'`.
            See :obj:`act_dict` for available types.
        data_shape : :obj:`list(int)`, optional
            Data shape to/from which output/input data is reshaped.
            See :attr:`data_shape` for details.

        Note
        ----
        :obj:`len(strides) == layer_sizes.shape[0] - 1`
        
        :obj:`len(paddings) == layer_sizes.shape[0] - 1`

        :obj:`len(dilations) == layer_sizes.shape[0] - 1`
        """
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
        """:obj:`int` : Depth of layers including input, hidden, output layers."""

        self.layer_sizes = layer_sizes
        """:obj:`numpy.array` : 2d integer array of shape :math:`[n\_layers, 3]`,
        indicating tensor dimension of each layer.
        For :math:`k`-th layer, the tensor dimension is

        .. math::
            layer\_sizes[k] = [channels, height, width]
        
        """

        self.channels = [layer_sizes[k][0] for k in range(self.n_layers)]
        """:obj:`list(int)` : list of channel size that
        determines architecture of each layer.
        For details on how architecture is determined,
        see `torch API documentation <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.
        """

        # assert(len(kernel_sizes) == self.n_layers - 1)
        assert(len(strides) == self.n_layers - 1)
        assert(len(paddings) == self.n_layers - 1)
        assert(len(dilations) == self.n_layers - 1)
        # self.kernel_sizes = kernel_sizes
        self.strides = strides
        """:obj:`list` : list of strides that
        determine architecture of each layer.
        Each stride can be either integer or tuple.
        For details on how architecture is determined,
        see `torch API documentation <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.
        """
        self.paddings = paddings
        """:obj:`list` : list of paddings that
        determine architecture of each layer.
        Each padding can be either integer or tuple.
        For details on how architecture is determined,
        see `torch API documentation <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.
        """
        self.dilations = dilations
        """:obj:`list` : list of dilations that
        determine architecture of each layer.
        Each dilation can be either integer or tuple.
        For details on how architecture is determined,
        see `torch API documentation <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.
        """

        self.groups = groups
        """:obj:`int` : groups that determine architecture of all layers.
        For details on how architecture is determined,
        see `torch API documentation <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.
        """
        self.bias = bias
        """:obj:`bool` : bias that determine architecture of all layers.
        For details on how architecture is determined,
        see `torch API documentation <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.
        """
        self.padding_mode = padding_mode
        """:obj:`str` : padding mode that determine architecture of all layers.
        For details on how architecture is determined,
        see `torch API documentation <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.
        """

        from lasdi.networks import act_dict
        # TODO(kevin): not use threshold activation for now.
        assert(act_type != 'threshold')
        self.act = act_dict[act_type]()
        """:obj:`torch.nn.Module` : activation function applied between all layers."""

        self.kernel_sizes = []
        """:obj:`list` : list of kernel_sizes that
        determine architecture of each layer.
        Each kernel_size can be either integer or tuple.
        Kernel size is automatically determined so that
        output of the corresponding layer has the shape of the next layer.

        For details on how architecture is determined,
        see `torch API documentation <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.
        """
        self.fcs = []
        """:obj:`torch.nn.ModuleList` : module list of
        :obj:`torch.nn.Conv2d` (forward) or :obj:`torch.nn.Conv2d` (backward)."""

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

        self.data_shape = data_shape
        """:obj:`list(int)` : tensor dimension of the training data
        that will be passed into/out of the module."""

        self.batch_reshape = None
        """:obj:`list(int)` : tensor dimension to which input/output data is reshaped.

        - Forward :attr:`mode`: shape of 3d-/4d-array
        - Backward :attr:`mode`: shape of arbitrary nd-array

        Determined by :meth:`set_data_shape`.
        """
        if (data_shape is not None):
            self.set_data_shape(data_shape)

        return
    
    def set_data_shape(self, data_shape : list):
        """
        Set the batch reshape in order to reshape the input/output batches
        based on given training data shape.

        Forward :attr:`mode`:

            For :obj:`data_shape` :math:`=[N_1,\ldots,N_m]`
            and the first layer size of :math:`[C_1, H_1, W_1]`,

            .. math::
                batch\_reshape = [R_1, C_1, H_1, W_1],

            where :math:`\prod_{i=1}^m N_i = R_1\\times C_1\\times H_1\\times W_1`.

            If :math:`m=2` and :math:`C_1=1`, then

            .. math::
                batch\_reshape = [C_1, H_1, W_1].

        Note
        ----
        For forward mode, :obj:`data_shape[-2:]==self.layer_sizes[0, 1:]` must be true.


        Backward :attr:`mode`:

            :attr:`batch_shape` is the same as :obj:`data_shape`.
            Output tensor of the module is reshaped as :obj:`data_shape`.

        Parameters
        ----------
        data_shape : :obj:`list(int)`
            Shape of the input/output data tensor for forward/backward mode.
        """
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

        self.data_shape = list(data_shape)
        return
    
    def print_data_shape(self):
        """Print out the data shape and architecture of the module."""

        mode_str = "forward" if (self.mode == CNN2D.Mode.Forward) else "backward"
        print("mode: ", mode_str)
        print("batch reshape: ", self.batch_reshape)
        for k in range(self.n_layers - 1):
            print('input layer: ', self.layer_sizes[k],
                  'kernel size: ', self.kernel_sizes[k],
                  'output layer: ', self.layer_sizes[k+1])
        return
    
    def forward(self, x):
        """Evaluate through the module.
        
        Parameters
        ----------
        x : :obj:`torch.nn.Tensor`
            Input tensor to pass into the module.

            - Forward mode: nd array of shape :attr:`data_shape`
            - Backward mode: Same shape as the output tensor of forward mode

        Returns
        -------
        :obj:`torch.nn.Tensor`
            Output tensor evaluated from the module.

            - Forward mode: 3d array of shape :obj:`self.layer_sizes[-1]`,
              or 4d array of shape :obj:`[self.batch_reshape[0]] + self.layer_sizes[-1]`
            - Backward mode: nd array of shape :attr:`data_shape` (equal to :attr:`batch_shape`)
        """
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
        """Compute kernel size that produces desired output shape from given input shape.

        The formula is based on torch API documentation
        for `Conv2d <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_
        and `ConvTranspose2d <https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html>`_.
        
        Parameters
        ----------
        input_shape : :obj:`int` or :obj:`tuple(int)`
        output_shape : :obj:`int` or :obj:`tuple(int)`
        stride : :obj:`int` or :obj:`tuple(int)`
        padding : :obj:`int` or :obj:`tuple(int)`
        dilation : :obj:`int` or :obj:`tuple(int)`
        mode : :class:`CNN2D.Mode`
            Direction of CNN. Either :attr:`CNN2D.Mode.Forward` or :attr:`CNN2D.Mode.Backward`

        Returns
        -------
        :obj:`list(int)`
            List of two integers indicating height and width of kernel.

        """
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
        """Compute input layer size that produces desired output shape with given kernel size.

        The formula is based on torch API documentation
        for `Conv2d <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_
        and `ConvTranspose2d <https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html>`_.
        
        Parameters
        ----------
        output_shape : :obj:`int` or :obj:`tuple(int)`
        kernel_size : :obj:`int` or :obj:`tuple(int)`
        stride : :obj:`int` or :obj:`tuple(int)`
        padding : :obj:`int` or :obj:`tuple(int)`
        dilation : :obj:`int` or :obj:`tuple(int)`
        mode : :class:`CNN2D.Mode`
            Direction of CNN. Either :attr:`CNN2D.Mode.Forward` or :attr:`CNN2D.Mode.Backward`

        Returns
        -------
        :obj:`list(int)`
            List of two integers indicating height and width of input layer.

        """
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
        """Compute output layer size produced from given input shape and kernel size.

        The formula is based on torch API documentation
        for `Conv2d <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_
        and `ConvTranspose2d <https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html>`_.
        
        Parameters
        ----------
        input_shape : :obj:`int` or :obj:`tuple(int)`
        kernel_size : :obj:`int` or :obj:`tuple(int)`
        stride : :obj:`int` or :obj:`tuple(int)`
        padding : :obj:`int` or :obj:`tuple(int)`
        dilation : :obj:`int` or :obj:`tuple(int)`
        mode : :class:`CNN2D.Mode`
            Direction of CNN. Either :attr:`CNN2D.Mode.Forward` or :attr:`CNN2D.Mode.Backward`

        Returns
        -------
        :obj:`list(int)`
            List of two integers indicating height and width of output layer.

        """
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
        """Initialize weights of linear features according to Xavier uniform distribution."""
        # TODO(kevin): support other initializations?
        for fc in self.fcs:
            torch.nn.init.xavier_uniform_(fc.weight)
        return