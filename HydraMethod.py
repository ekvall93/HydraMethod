"""Markus Ekvall: 2018-12-05."""
from keras.models import Model
from keras.losses import mean_absolute_error
from keras.optimizers import Adam, SGD
from keras.layers import Input, concatenate, Dense
from keras.preprocessing.sequence import pad_sequences


class HydraMethod():
    """Hydra method for 1 dimensional data."""

    def __init__(self, architecture, nr_heads, new_architecture=True):
        """
        :param architecture: Architecture to predict 1D sequences.
        :param nr_heads: Number of heads applied to architecture.
        :returns: Trained model.
        """
        self.architecture = architecture
        self.nr_heads = nr_heads
        self.new_architecture = new_architecture

    def get_architecture(self, nr_heads):
        """
        Get architecture with the Hydra method.

        :param nr_heads: Number of heads applied to architecture.
        :returns: Compiled model.
        """
        d = {}
        X, X_input = list(), list()
        for h in range(nr_heads):
            d["x_{0}".format(h)] = Input(shape=(32,), name="x_{0}".format(h))
            X_input += [d["x_{0}".format(h)]]
            d["ResnetRnnDense_{0}".format(h)] = self.architecture
            d["x_{0}".format(h)] = d["ResnetRnnDense_{0}".format(h)].compile(
                                                                    X_input[h])
            X += [d["x_{0}".format(h)]]
        if nr_heads > 1:
            x_c = concatenate(X, axis=1)
            y = Dense(1)(x_c)
        else:
            y = X[h]
        return Model(inputs=X_input, outputs=y)

    def reverse_sequences(self, x):
        """
        Reverse the 1 dimensional sequence.

        :param sequence: Number of heads applied to architecture.
        :returns: Reversed sequence.
        """
        backward_x = list()
        for s in x:
            backward_x += [s[::-1]]
        return backward_x

    def get_representation(self, head, x):
        """
        Get different representation of 1D sequence.

        :param head: What head i.e. get input to correct head of the model.
        :param x: 1 dimensional sequence.
        :returns: represnetation of 1 dimensional sequence.
        """
        if head == 0 or head == 2:
            if head == 0:
                return pad_sequences(self.forward(x), maxlen=32,
                                     padding="pre")
            elif head == 2:
                return pad_sequences(self.forward(x), maxlen=32,
                                     padding="post")
        elif head == 1 or head == 3:
            if head == 1:
                return pad_sequences(self.backward(x), maxlen=32,
                                     padding="pre")
            elif head == 3:
                return pad_sequences(self.backward(x), maxlen=32,
                                     padding="post")

    def forward(self, x):
        """
        Forward direction of sequence.

        :param x: Seqeucne.
        :returns: Forward seqeucne.
        """
        return x

    def backward(self, x):
        """
        Backward direction of sequence.

        :param x: Seqeucne.
        :returns: Backward seqeucne.
        """
        return self.reverse_sequences(x)

    def get_all_representation(self, x):
        """
        Get all different types of sequence representation.

        :param x: Seqeucne.
        :param nr_heads: Number of heads used.
        :returns: Severeal representations of sequence.
        """
        X = list()
        for h in range(self.nr_heads):
            X += [self.get_representation(h, x)]
        return X

    def compile(self, optimizer=Adam, loss=mean_absolute_error):
        """
        Compile model.

        :param optimizer: Optimzer.
        :param loss: Optimization function.
        :param learning_rate: learning_rate.
        :returns: Severeal representations of sequence.
        """
        if self.new_architecture:
            self.model = self.get_architecture(self.nr_heads)
            self.new_architecture = False
        self.model.compile(loss=loss, optimizer=optimizer)

    def get_model(self):
        """
        Compile model.

        :param weight_name: Name for weight name file.
        :returns: Severeal representations of sequence.
        """
        return self.model

    def fit(self, x_train, y_train, x_test, y_test, epochs=3, batch_size=64):
        """
        Fit model with the Hydra method.

        :param x_train: Seqeucne training data.
        :param y_train: Training target data.
        :param x_test: Seqeucne test data.
        :param y_test: Training test data.
        :returns: trained model
        """
        X_train = self.get_all_representation(x_train)
        X_test = self.get_all_representation(x_test)
        self.model.fit(X_train, y_train, validation_data=(
                        X_test, y_test), epochs=epochs, batch_size=batch_size)
        return self.model
