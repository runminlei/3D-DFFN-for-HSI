import six
from keras.models import Model
import keras
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
    Concatenate,
    GlobalAveragePooling3D)
from keras.layers.convolutional import (
    Convolution3D,
    MaxPooling3D,
    AveragePooling3D,
    Conv3D,
    Conv2D
)
from keras import backend as K
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.layers.core import Reshape
from keras import regularizers
from keras.layers.merge import add
from keras.utils import plot_model

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    W_regularizer = conv_params.setdefault("W_regularizer", regularizers.l2(1.e-4))

    def f(input):
        conv = Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer, filters=nb_filter,
                      kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3))(input)

        return _bn_relu(conv)

    return f

def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    """
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                      filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3),
                      padding=border_mode)(activation)

    return f

def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    stride_dim1 = (input._keras_shape[CONV_DIM1] + 1) // residual._keras_shape[CONV_DIM1]
    stride_dim2 = (input._keras_shape[CONV_DIM2] + 1) // residual._keras_shape[CONV_DIM2]
    stride_dim3 = (input._keras_shape[CONV_DIM3] + 1) // residual._keras_shape[CONV_DIM3]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input

    # 1 X 1 conv if shape is different. Else identity.
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
        shortcut = Convolution3D(nb_filter=residual._keras_shape[CHANNEL_AXIS],
                                 kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
                                 subsample=(stride_dim1, stride_dim2, stride_dim3),
                                 init="he_normal", border_mode="valid",
                                 W_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

def _residual_block(block_function, nb_filter, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        print('repetitions:', repetitions)

        for i in range(repetitions):
            init_subsample = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (1, 1, 2)
            input = block_function(
                nb_filter=nb_filter,
                init_subsample=init_subsample,
                is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f

def _residual_block_(block_function, nb_filter, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        print('repetitions:', repetitions)

        for i in range(repetitions):
            init_subsample = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (1, 1, 2)
            input = basic_block_(
                nb_filter=nb_filter,
                init_subsample=init_subsample,
                is_first_block_of_first_layer=is_first_layer)(input)
        return input

    return f

def basic_block_(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Basic 1x1 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """

    # 如果是第一层的话，就进入一个Conv3D,否则进行_bn_relu_conv = BN + RELU + CONV,最后再走一遍
    # BN + RELU + CONV,实现一个shortcut

    def f(input):


        conv1 = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3,
                                  subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3)(conv1)
        return _shortcut(input, residual)

    return f

def basic_block(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Basic 1x1 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """

    # 如果是第一层的话，就进入一个Conv3D,否则进行_bn_relu_conv = BN + RELU + CONV,最后再走一遍
    # BN + RELU + CONV,实现一个shortcut

    def f(input):

        if is_first_block_of_first_layer:
            conv1 = Conv3D(kernel_initializer="he_normal", strides=init_subsample,
                           kernel_regularizer=regularizers.l2(0.0001),
                           filters=nb_filter, kernel_size=(3, 3, 3), padding='same')(input)
        else:
            conv1 = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3,
                                  subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3)(conv1)
        return _shortcut(input, residual)

    return f

def _handle_dim_ordering():
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        CONV_DIM1 = 1
        CONV_DIM2 = 2
        CONV_DIM3 = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1 = 2
        CONV_DIM2 = 3
        CONV_DIM3 = 4

def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier

class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        print('original input shape:', input_shape)
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        print('original input shape:', input_shape)

        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])
        print('change input shape:', input_shape)

        # Load function from str if needed.
        # basic_block = 残差模块
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)

        # Conv3D + BN + Relu
        conv1 = _conv_bn_relu(nb_filter=24, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, subsample=(1, 1, 2))(
            input)

        block1 = conv1
        nb_filter = 16
        for i, r in enumerate(repetitions):
            block1 = _residual_block(block_fn, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(
                block1)
            nb_filter *= 1

        # print("1" * 10, block1.shape)

        block2 = block1
        nb_filter = 32
        for i, r in enumerate(repetitions):
            block2 = _residual_block_(block_fn, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(
                block2)
            nb_filter *= 1
        # print("2" * 10, block2.shape)

        block3 = block2
        nb_filter = 64
        for i, r in enumerate(repetitions):
            block3 = _residual_block_(block_fn, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(
                block3)
            nb_filter *= 1
        # print("3" * 10, block3.shape)

        # 聚合视图
        block1 = Conv3D(64, kernel_size=(1, 1, 1), padding='SAME', kernel_initializer='he_normal')(
            block1)
        block2 = Conv3D(64, kernel_size=(1, 1, 1), padding='SAME', kernel_initializer='he_normal')(
            block2)
        block_ss = keras.layers.add([block1, block2, block3])

        # 两层 BN 加强正则化
        # Last activation
        block_norm_spc = BatchNormalization(axis=CHANNEL_AXIS)(block_ss)
        block_output_spc = Activation("relu")(block_norm_spc)

        block = _bn_relu(block_output_spc)

        # Classifier block
        pool2 = GlobalAveragePooling3D(name='avg_pool')(block)
        dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(pool2)

        block_norm_spc_1 = BatchNormalization(axis=CHANNEL_AXIS)(block3)
        block_output_spc_1 = Activation("relu")(block_norm_spc_1)

        block_1 = _bn_relu(block_output_spc_1)

        pool2_1 = GlobalAveragePooling3D(name='avg_pool_1')(block_1)
        den = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(pool2_1)

        # model = Model(inputs=input, outputs=[dense, den])
        train_model = Model(inputs=input, outputs=[dense, den])
        eval_model = Model(inputs=input, outputs=dense)
        return train_model, eval_model

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3])

def main():
    model, eval_model = ResnetBuilder.build_resnet_8((1, 19, 19, 200), 16)
    model.compile(loss=["categorical_crossentropy", "categorical_crossentropy"],
                  loss_weights=[1., 0.5], optimizer="sgd")
    model.summary(positions=[.33, .61, .71, 1.])

    # Save a PNG of the Model Build
    # plot_model(model, to_file='DFFN_1.png')

if __name__ == '__main__':
    main()