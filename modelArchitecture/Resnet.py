"""Markus Ekvall: 2018-12-05."""
from keras.layers import ZeroPadding1D, Conv1D, Activation, MaxPooling1D, Add
from .VirtualBatchNormalization import VirtualBatchNormalization
from keras.layers import BatchNormalization

class Resnet():
    """Resnet Architecture
    Credit: Allen Goodman
    https://github.com/broadinstitute/keras-resnet
    """
    def __init__(self, blocks, features, kernel_size=3,
                 virtual_batch_size=None):
        """
        :param blocks: Number of ResNet blocks.
        :param features: Features/Filters
        :param kernel_size: kernel/window
        """
        self.blocks = blocks
        self.features = features
        self.kernel_size = kernel_size
        self.virtual_batch_size = virtual_batch_size

    def resnet_block(self, filters, stage=0, block=0, stride=None,
                     numerical_name=False):
        """
        A one-dimensional basic block.

        :param filters: the output’s feature space.
        :param stage: int representing the stage of this block.
        :param block: int representing this block.
        :param numerical_name: Numbers to represent blocks.
        :param stride: Sride used in the shortcut and the first conv layer.

        :returns: Resnet block.
        """

        parameters = {
            "kernel_initializer": "he_normal"
        }

        if stride is None:
            if block != 0 or stage == 0:
                stride = 1
            else:
                stride = 2

        if block > 0 and numerical_name:
            block_char = "b{}".format(block)
        else:
            block_char = chr(ord('a') + block)

        stage_char = str(stage + 2)

        def f(x):
            """
            ResNet block

            :param x: Acitvaitions.
            :returns: ResnetBlock activations.
            """
            y = ZeroPadding1D(padding=1)(x)
            y = Conv1D(filters, self.kernel_size, strides=stride,
                       use_bias=False, **parameters)(y)
            #y = VirtualBatchNormalization(
            #           virtual_batch_size=self.virtual_batch_size)(y)
            y = BatchNormalization()(y)
            y = Activation("elu")(y)

            y = ZeroPadding1D(padding=1)(y)
            y = Conv1D(filters, self.kernel_size, use_bias=False,
                       **parameters)(y)
            #y = VirtualBatchNormalization(
            #           virtual_batch_size=self.virtual_batch_size)(y)
            y = BatchNormalization()(y)
            if block == 0:
                shortcut = Conv1D(filters, 1, strides=stride, use_bias=False,
                                  **parameters)(x)
                #shortcut = VirtualBatchNormalization(
                #          virtual_batch_size=self.virtual_batch_size)(shortcut)
                shortcut = BatchNormalization()(shortcut)
            else:
                shortcut = x

            y = Add()([y, shortcut])
            y = Activation("elu")(y)
            return y
        return f

    def resnet(self, x, init_maxpool=False):
        """
        Get ResNet.

        :param x: Input.
        :param init_maxpool: Maxpool or Conv downsample.
        :returns: ResNet Architecture.
        """

        x = ZeroPadding1D(padding=3)(x)
        x = Conv1D(64, 7, strides=2, use_bias=False)(x)
        #x = VirtualBatchNormalization(
        #    virtual_batch_size=self.virtual_batch_size
        #)(x)
        x = BatchNormalization()(x)
        x = Activation("elu")(x)

        if init_maxpool:
            x = MaxPooling1D(3, strides=2, padding="same")(x)
        else:
            x = Conv1D(64, 3, strides=2, use_bias=False)(x)
            #x = VirtualBatchNormalization(
            #    virtual_batch_size=self.virtual_batch_size)(x)
            x = BatchNormalization()(x)
            x = Activation("elu")(x)
        for stage_id, iterations in enumerate(self.blocks):
            for block_id in range(iterations):
                x = self.resnet_block(
                    self.features, stage_id, block_id)(x)
            self.features *= 2

        return x
