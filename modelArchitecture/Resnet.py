from keras.layers import ZeroPadding1D, Conv1D, Activation, MaxPooling1D,Add
from .BatchNormalization import BatchNormalization

class Resnet():
    def __init__(self, blocks, features,
                 kernel_size=3, freeze_bn=False,*args, **kwargs):
        self.blocks = blocks
        self.features = features
        self.freeze_bn = freeze_bn
        self.kernel_size = kernel_size
        super(Resnet, self).__init__(*args, **kwargs)

        # set to non-trainable if freeze is true


    def basic_1d(self, filters, stage=0, block=0, stride=None, numerical_name=False):
        """
        A one-dimensional basic block.
        :param filters: the output’s feature space
        :param stage: int representing the stage of this block (starting from 0)
        :param block: int representing this block (starting from 0)
        :param kernel_size: size of the kernel
        :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
        :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
        :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
        Usage:
            >>> import keras_resnet.blocks
            >>> keras_resnet.blocks.basic_1d(64)
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
            y = ZeroPadding1D(padding=1)(x)
            y = Conv1D(filters, self.kernel_size,
                                    strides=stride, use_bias=False,
                                    **parameters)(y)
            y = BatchNormalization(axis=1, epsilon=1e-5, freeze=self.freeze_bn)(y)
            y = Activation("elu")(y)

            y = ZeroPadding1D(padding=1)(y)
            y = Conv1D(filters, self.kernel_size, use_bias=False,
                                    **parameters)(y)
            y = BatchNormalization(axis=1, epsilon=1e-5,
                                   freeze=self.freeze_bn)(y)
            if block == 0:
                shortcut = Conv1D(filters, 1, strides=stride,
                                               use_bias=False, **parameters)(x)
                shortcut = BatchNormalization(axis=1, epsilon=1e-5,
                                              freeze=self.freeze_bn)(shortcut)
            else:
                shortcut = x

            y = Add()([y, shortcut])
            y = Activation("elu")(y)

            return y

        return f

    def resnet(self, x, init_maxpool=False):

        x = ZeroPadding1D(padding=3)(x)
        x = Conv1D(64, 7, strides=2, use_bias=False )(x)
        x = BatchNormalization(axis=1, epsilon=1e-5, freeze=self.freeze_bn)(x)
        x = Activation("elu")(x)

        if init_maxpool:
            x = MaxPooling1D(3, strides=2, padding="same")(x)
        else:
            x = Conv1D(64, 3, strides=2, use_bias=False )(x)
            x = BatchNormalization(axis=1, epsilon=1e-5, freeze=self.freeze_bn)(x)
            x = Activation("elu")(x)



        for stage_id, iterations in enumerate(self.blocks):
            for block_id in range(iterations):
                x = self.basic_1d(self.features, stage_id, block_id,
                             numerical_name=(block_id > 0 and
                                             numerical_names[stage_id]))(x)
            self.features *= 2

        return x
