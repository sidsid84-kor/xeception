
import torch
from torch import nn
from torch.nn import functional as F
import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo



# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
    'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

# Set GlobalParams and BlockArgs's defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

# Swish activation function
if hasattr(nn, 'SiLU'):
    Swish = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def round_filters(filters, global_params):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor and min_depth of global_params.

    Args:
        filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.

    Returns:
        new_filters: New filters number after calculating.
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    # TODO: modify the params names.
    #       maybe the names (width_divisor,min_width)
    #       are more suitable than (depth_divisor,min_depth).
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor  # pay attention to this line when using min_depth
    # follow the formula transferred from official TensorFlow implementation
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Calculate module's repeat number of a block based on depth multiplier.
       Use depth_coefficient of global_params.

    Args:
        repeats (int): num_repeat to be calculated.
        global_params (namedtuple): Global params of the model.

    Returns:
        new repeat: New repeat number after calculating.
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """Drop connect.

    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.

    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


def get_width_and_height_from_size(x):
    """Obtain height and width from x.

    Args:
        x (int, tuple or list): Data size.

    Returns:
        size: A tuple or list (H,W).
    """
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.

    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.

    Returns:
        output_image_size: A list [H,W].
    """
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


# Note:
# The following 'SamePadding' functions make output size equal ceil(input size/stride).
# Only when stride equals 1, can the output size be the same as input size.
# Don't be confused by their function names ! ! !

def get_same_padding_conv2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.

    Args:
        image_size (int or tuple): Size of the image.

    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
    """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


def get_same_padding_maxPool2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.

    Args:
        image_size (int or tuple): Size of the image.

    Returns:
        MaxPool2dDynamicSamePadding or MaxPool2dStaticSamePadding.
    """
    if image_size is None:
        return MaxPool2dDynamicSamePadding
    else:
        return partial(MaxPool2dStaticSamePadding, image_size=image_size)


class MaxPool2dDynamicSamePadding(nn.MaxPool2d):
    """2D MaxPooling like TensorFlow's 'SAME' mode, with a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    def __init__(self, kernel_size, stride, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding,
                            self.dilation, self.ceil_mode, self.return_indices)


class MaxPool2dStaticSamePadding(nn.MaxPool2d):
    """2D MaxPooling like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    def __init__(self, kernel_size, stride, image_size=None, **kwargs):
        super().__init__(kernel_size, stride, **kwargs)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding,
                         self.dilation, self.ceil_mode, self.return_indices)
        return x


################################################################################
# Helper functions for loading model params
################################################################################

# BlockDecoder: A Class for encoding and decoding BlockArgs
# efficientnet_params: A function to query compound coefficient
# get_model_params and efficientnet:
#     Functions to get BlockArgs and GlobalParams for efficientnet
# load_pretrained_weights: A function to load pretrained weights

class BlockDecoder(object):
    """Block Decoder for readability,
       straight from the official TensorFlow repository.
    """

    @staticmethod
    def _decode_block_string(block_string):
        """Get a block through a string notation of arguments.

        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.

        Returns:
            BlockArgs: The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            num_repeat=int(options['r']),
            kernel_size=int(options['k']),
            stride=[int(options['s'][0])],
            expand_ratio=int(options['e']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            se_ratio=float(options['se']) if 'se' in options else None,
            id_skip=('noskip' not in block_string))

    @staticmethod
    def _encode_block_string(block):
        """Encode a block to a string.

        Args:
            block (namedtuple): A BlockArgs type argument.

        Returns:
            block_string: A String form of BlockArgs.
        """
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """Decode a list of string notations to specify blocks inside the network.

        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.

        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """Encode a list of BlockArgs to a list of strings.

        Args:
            blocks_args (list[namedtuples]): A list of BlockArgs namedtuples of block args.

        Returns:
            block_strings: A list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet_params(model_name):
    """Map EfficientNet model name to parameter coefficients.

    Args:
        model_name (str): Model name to be queried.

    Returns:
        params_dict[model_name]: A (width,depth,res,dropout) tuple.
    """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
        'efficientnet-c1': (2.2, 1.0, 640, 0.2),
        
    }
    return params_dict[model_name]


def efficientnet(width_coefficient=None, depth_coefficient=None, image_size=None,
                 dropout_rate=0.2, drop_connect_rate=0.2, num_classes=1000, include_top=True):
    """Create BlockArgs and GlobalParams for efficientnet model.

    Args:
        width_coefficient (float)
        depth_coefficient (float)
        image_size (int)
        dropout_rate (float)
        drop_connect_rate (float)
        num_classes (int)

        Meaning as the name suggests.

    Returns:
        blocks_args, global_params.
    """

    # Blocks args for the whole model(efficientnet-b0 by default)
    # It will be modified in the construction of EfficientNet Class according to model
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        image_size=image_size,
        dropout_rate=dropout_rate,

        num_classes=num_classes,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        drop_connect_rate=drop_connect_rate,
        depth_divisor=8,
        min_depth=None,
        include_top=include_top,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model name.

    Args:
        model_name (str): Model's name.
        override_params (dict): A dict to modify global_params.

    Returns:
        blocks_args, global_params
    """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: {}'.format(model_name))
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8','efficientnet-c1',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)


class MBConvBlock(nn.Module):
    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class TH_EfficientNet(nn.Module):

    def __init__(self, num_classes = 6):
        super().__init__()

        self._global_params = GlobalParams(
            width_coefficient=2.2, depth_coefficient=1.0, image_size=640, 
            dropout_rate=0.2, num_classes=num_classes, batch_norm_momentum=0.99, 
            batch_norm_epsilon=0.001, drop_connect_rate=0.2, depth_divisor=8, 
            min_depth=None, include_top=True
        )

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = self._global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        self.MBblock_1 = MBConvBlock(
            BlockArgs(num_repeat=1, kernel_size=3, stride=[1], expand_ratio=1, input_filters=72, output_filters=32, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )

        self.MBblock_2_1 = MBConvBlock(
            BlockArgs(num_repeat=2, kernel_size=3, stride=[2], expand_ratio=6, input_filters=32, output_filters=56, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        image_size = [160, 160] # = calculate_output_image_size(image_size, 이전 stride)

        self.MBblock_2_2 = MBConvBlock(
            BlockArgs(num_repeat=2, kernel_size=3, stride=1, expand_ratio=6, input_filters=56, output_filters=56, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )

        self.no_etc_output_linear = nn.Linear(56, 2)

        self.MBblock_3_1 = MBConvBlock(
            BlockArgs(num_repeat=2, kernel_size=5, stride=[2], expand_ratio=6, input_filters=56, output_filters=88, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        self.MBblock_3_1_1 = MBConvBlock(
            BlockArgs(num_repeat=2, kernel_size=5, stride=[2], expand_ratio=6, input_filters=56, output_filters=88, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        image_size = [80, 80] # = calculate_output_image_size(image_size, 이전 stride [2])

        self.MBblock_3_2 = MBConvBlock(
            BlockArgs(num_repeat=2, kernel_size=5, stride=1, expand_ratio=6, input_filters=88, output_filters=88, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        

        self.MBblock_4_1 = MBConvBlock(
            BlockArgs(num_repeat=3, kernel_size=3, stride=[2], expand_ratio=6, input_filters=88, output_filters=176, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )

        self.MBblock_3_2_1 = MBConvBlock(
            BlockArgs(num_repeat=2, kernel_size=5, stride=1, expand_ratio=6, input_filters=88, output_filters=88, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        

        self.MBblock_4_1_1 = MBConvBlock(
            BlockArgs(num_repeat=3, kernel_size=3, stride=[2], expand_ratio=6, input_filters=88, output_filters=176, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        image_size = [40, 40] # = calculate_output_image_size(image_size, 이전 stride)

        self.MBblock_4_2 = MBConvBlock(
            BlockArgs(num_repeat=3, kernel_size=3, stride=1, expand_ratio=6, input_filters=176, output_filters=176, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        self.MBblock_4_3 = MBConvBlock(
            BlockArgs(num_repeat=3, kernel_size=3, stride=1, expand_ratio=6, input_filters=176, output_filters=176, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        

        self.MBblock_5_1 = MBConvBlock(
            BlockArgs(num_repeat=3, kernel_size=5, stride=[1], expand_ratio=6, input_filters=176, output_filters=248, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )

        self.MBblock_5_2 = MBConvBlock(
            BlockArgs(num_repeat=3, kernel_size=5, stride=1, expand_ratio=6, input_filters=248, output_filters=248, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        self.MBblock_5_3 = MBConvBlock(
            BlockArgs(num_repeat=3, kernel_size=5, stride=1, expand_ratio=6, input_filters=248, output_filters=248, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        
        self.burr_broken_output_linear = nn.Linear(176, 2)

        self.MBblock_6_1 = MBConvBlock(
            BlockArgs(num_repeat=4, kernel_size=5, stride=[2], expand_ratio=6, input_filters=248, output_filters=424, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )

        self.MBblock_4_2_1 = MBConvBlock(
            BlockArgs(num_repeat=3, kernel_size=3, stride=1, expand_ratio=6, input_filters=176, output_filters=176, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        self.MBblock_4_3_1 = MBConvBlock(
            BlockArgs(num_repeat=3, kernel_size=3, stride=1, expand_ratio=6, input_filters=176, output_filters=176, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        

        self.MBblock_5_1_1 = MBConvBlock(
            BlockArgs(num_repeat=3, kernel_size=5, stride=[1], expand_ratio=6, input_filters=176, output_filters=248, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )

        self.MBblock_5_2_1 = MBConvBlock(
            BlockArgs(num_repeat=3, kernel_size=5, stride=1, expand_ratio=6, input_filters=248, output_filters=248, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        self.MBblock_5_3_1 = MBConvBlock(
            BlockArgs(num_repeat=3, kernel_size=5, stride=1, expand_ratio=6, input_filters=248, output_filters=248, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        

        self.MBblock_6_1_1 = MBConvBlock(
            BlockArgs(num_repeat=4, kernel_size=5, stride=[2], expand_ratio=6, input_filters=248, output_filters=424, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        image_size = [20, 20] # = calculate_output_image_size(image_size, 이전 stride)

        self.MBblock_6_2 = MBConvBlock(
            BlockArgs(num_repeat=4, kernel_size=5, stride=1, expand_ratio=6, input_filters=424, output_filters=424, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        self.MBblock_6_3 = MBConvBlock(
            BlockArgs(num_repeat=4, kernel_size=5, stride=1, expand_ratio=6, input_filters=424, output_filters=424, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        self.MBblock_6_4 = MBConvBlock(
            BlockArgs(num_repeat=4, kernel_size=5, stride=1, expand_ratio=6, input_filters=424, output_filters=424, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        

        self.MBblock_7 = MBConvBlock(
            BlockArgs(num_repeat=1, kernel_size=3, stride=[1], expand_ratio=6, input_filters=424, output_filters=704, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        self.b_edge_linear = nn.Linear(2816, 1)

        self.MBblock_6_2_1 = MBConvBlock(
            BlockArgs(num_repeat=4, kernel_size=5, stride=1, expand_ratio=6, input_filters=424, output_filters=424, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        self.MBblock_6_3_1 = MBConvBlock(
            BlockArgs(num_repeat=4, kernel_size=5, stride=1, expand_ratio=6, input_filters=424, output_filters=424, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        self.MBblock_6_4_1 = MBConvBlock(
            BlockArgs(num_repeat=4, kernel_size=5, stride=1, expand_ratio=6, input_filters=424, output_filters=424, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        

        self.MBblock_7_1 = MBConvBlock(
            BlockArgs(num_repeat=1, kernel_size=3, stride=[1], expand_ratio=6, input_filters=424, output_filters=704, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )

        # Head
        in_channels = 704 #block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._conv_head_1 = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1_1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            # self._fc = nn.Linear(out_channels, self._global_params.num_classes)
            self._fc = nn.Linear(out_channels, 1)

        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
                >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints


    def forward(self, inputs):
        outputs = []
        x1 = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        x2 = self.MBblock_1(x1)

        x3 = self.MBblock_2_1(x2)
        x4 = self.MBblock_2_2(x3)

        no_etc_output = self._avg_pooling(x4)
        no_etc_output = torch.flatten(no_etc_output, 1)
        no_etc_output = self.no_etc_output_linear(no_etc_output)
        outputs.append(no_etc_output)

        x5 = self.MBblock_3_1(x4)
        x6 = self.MBblock_3_2(x5)

        x7 = self.MBblock_4_1(x6)
        x8 = self.MBblock_4_2(x7)
        x9 = self.MBblock_4_3(x8)

        burr_broken_output = self._avg_pooling(x9)
        burr_broken_output = torch.flatten(burr_broken_output, 1)
        burr_broken_output = self.burr_broken_output_linear(burr_broken_output)
        outputs.append(burr_broken_output)

        x10 = self.MBblock_5_1(x9)
        x11 = self.MBblock_5_2(x10)
        x12 = self.MBblock_5_3(x11)

        x13 = self.MBblock_6_1(x12)
        #x14 = self.MBblock_6_2(x13)
        #x15 = self.MBblock_6_3(x14)
        x16 = self.MBblock_6_4(x13)

        x17 = self.MBblock_7(x16)

        # Head
        x18 = self._swish(self._bn1(self._conv_head(x17)))
        # Pooling and final linear layer
        b_edge_output = self._avg_pooling(x18)
        b_edge_output = torch.flatten(b_edge_output, 1)
        #b_edge_output = self._dropout(b_edge_output)
        b_edge_output = self.b_edge_linear(b_edge_output)
        outputs.append(b_edge_output)

        x5_1 = self.MBblock_3_1_1(x4)
        x6_1 = self.MBblock_3_2_1(x5_1)

        x7_1 = self.MBblock_4_1_1(x6_1)
        x8_1 = self.MBblock_4_2_1(x7_1)
        x9_1 = self.MBblock_4_3_1(x8_1)

        x10_1 = self.MBblock_5_1_1(x9_1)
        x11_1 = self.MBblock_5_2_1(x10_1)
        x12_1 = self.MBblock_5_3_1(x11_1)

        x13_1 = self.MBblock_6_1_1(x12_1)
        #x14_1 = self.MBblock_6_2_1(x13_1)
        #x15_1 = self.MBblock_6_3_1(x14_1)
        x16_1 = self.MBblock_6_4_1(x13_1)

        x17_1 = self.MBblock_7_1(x16_1)

        # Head
        x18_1 = self._swish(self._bn1_1(self._conv_head_1(x17_1)))
        b_bubble_output = self._avg_pooling(x18_1)
        b_bubble_output = torch.flatten(b_bubble_output, 1)
        #b_bubble_output = self._dropout(b_bubble_output)
        b_bubble_output = self._fc(b_bubble_output)
        outputs.append(b_bubble_output)

        final_outputs = torch.cat(outputs, dim=1)

        return final_outputs


