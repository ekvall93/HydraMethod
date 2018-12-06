"""Markus Ekvall: 2018-12-05."""
from .Resnet import Resnet
from .Attention import Attention
from keras.layers import (Embedding, Input, Bidirectional, CuDNNLSTM, Dense,
                          concatenate, SpatialDropout1D, Dropout)


class ResnetRnnDense():
    """ResnetRnnDense Architecture"""
    def __init__(self, RnnDrop=0.5, DenseDrop=0.5):
        """
        :param RnnDrop: Droput after RNN.
        :param DenseDrop: Droput after Dense.
        """
        self.RnnDrop = RnnDrop
        self.DenseDrop = DenseDrop

    def ResNet(self, x):
        """
        Get ResNet Architecture

        :param x: Input activations.
        :returns: ResNet activations.
        """
        resnet10 = Resnet([1, 1, 1, 1], 64)
        return resnet10.resnet(x)

    def RNN(self, x, k):
        """
        Get RNN Architecture

        :param x: Input activations.
        :param k: Features.
        :returns: RNN Architecture
        """
        x = Bidirectional(CuDNNLSTM(k, return_sequences=True))(x)
        x = Attention()(x)
        return Dropout(self.RnnDrop)(x)

    def DeseNet(self, x, k):
        """
        Get DenseNet Architecture

        :param x: Input activations.
        :param k: Features.
        :returns: DenseNet Architecture
        """
        x = Dense(k, activation="elu")(x)
        return Dropout(self.DenseDrop)(x)

    def compile(self, x):
        """
        Compile ResnetRnnDense Architecture

        :param x: Input activations.
        :returns: ResnetRnnDense Architecture
        """
        x = Embedding(22, 32, input_length=32)(x)

        x = self.ResNet(x)

        k = int(x.shape[2])

        x = self.RNN(x, k)

        x = self.DeseNet(x, k)

        return Dense(1, activation="relu")(x)
