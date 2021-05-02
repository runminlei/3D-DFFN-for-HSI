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
from Utils.non_local import non_local_block

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

def dense_block(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x

def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv3D(4 * growth_rate, 1, use_bias=False,
                name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv3D(growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs):
        print('original input shape:', input_shape)
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        print('original input shape:', input_shape)
        # orignal input shape: 1,7,7,200

        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])
        print('change input shape:', input_shape)

        input = Input(shape=input_shape)

        # 3D Convolution and pooling
        conv1 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME', kernel_initializer='he_normal')(
            input)
        pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2))(conv1)

        bn_axis = 4 if K.image_data_format() == 'channels_last' else 1

        x1 = dense_block(pool1, 3, name='conv1')
        #x2 = non_local_block(x1, mode='embedded', compression=2)
        x3 = dense_block(x1, 3, name='conv2')
        #x4 = non_local_block(x3, mode='embedded', compression=2)
        x5 = dense_block(x3, 3, name='conv3')
        #x6 = non_local_block(x5, mode='embedded', compression=2)
        x7 = Concatenate(axis=bn_axis)([x1, x3, x5])

        x = GlobalAveragePooling3D(name='avg_pool')(x7)

        # 输入分类器
        dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(x)
        # dense1 = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(GlobalAveragePooling3D()(x3))
        den = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal", name='auxiliary')(GlobalAveragePooling3D()(x5))

        # model = Model(inputs=input, outputs=dense)
        train_model = Model(inputs=input, outputs=[dense, den])
        eval_model = Model(inputs=input, outputs=dense)
        return train_model, eval_model

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs)

# def main():
#     model = ResnetBuilder.build_resnet_8((1, 11, 11, 200), 16)
#     model.compile(loss="categorical_crossentropy", optimizer="sgd")
#     model.summary(positions=[.33, .61, .71, 1.])
    # plot_model(model, to_file='DFAN_Dense.png')

def main():
    model, eval_model = ResnetBuilder.build_resnet_8((1, 11, 11, 200), 16)
    model.compile(loss=["categorical_crossentropy", "categorical_crossentropy"],
                  loss_weights=[1., 0.5],optimizer="sgd")
    model.summary(positions=[.33, .61, .71, 1.])
    print(model.metrics_names)

if __name__ == '__main__':
    main()
