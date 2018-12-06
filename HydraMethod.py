from keras.models import Model
from keras.losses import mean_absolute_error
from keras.optimizers import Adam
from keras.layers import Input, concatenate, Dense
from keras.preprocessing.sequence import pad_sequences

class HydraMethod():
    def __init__(self,architecture,nr_heads):
        self.architecture = architecture
        self.nr_heads = nr_heads


    def compile_model(self, nr_heads):
        d={}
        X, X_input = list(), list()
        for h in range(nr_heads):
            d["x_{0}".format(h)] = Input(shape=(32,), name="x_{0}".format(h))
            X_input += [d["x_{0}".format(h)]]
            d["ResnetRnnDense_{0}".format(h)] = self.architecture()
            d["x_{0}".format(h)] = d["ResnetRnnDense_{0}".format(h)].compile(X_input[h])
            X += [d["x_{0}".format(h)]]
        if nr_heads > 1:
            x_c = concatenate(X, axis=1)
            y = Dense(1)(x_c)
        else:
            y = Dense(1)(X[h])

        return Model(inputs=X_input, outputs=y)

    def reverse_sequences(self, sequence):
        backward_seq = list()
        for seq in sequence:
            backward_seq += [seq[::-1]]
        return backward_seq

    def get_representation(self, h, x):
        if h == 0 or h == 2:
            if h == 0:
                return pad_sequences(self.forward(x), maxlen=32, padding="pre")
            elif h == 2:
                return pad_sequences(self.forward(x), maxlen=32, padding="post")
        elif h == 1 or h == 3:
            if h == 1:
                return pad_sequences(self.backward(x), maxlen=32, padding="pre")
            elif h == 3:
                return pad_sequences(self.backward(x), maxlen=32, padding="post")

    def forward(self,x):
        return x

    def backward(self,x):
        return self.reverse_sequences(x)


    def get_all_representation(self, x, nr_heads):
        X = list()
        for h in range(nr_heads):
            X +=[self.get_representation(h, x)]
        return X


    def fit(self, x_train, y_train, x_test, y_test):
        X_train= self.get_all_representation(x_train, self.nr_heads)
        X_test= self.get_all_representation(x_test, self.nr_heads)

        model = self.compile_model(self.nr_heads)
        model.compile(loss = mean_absolute_error,
                      optimizer = Adam(1e-4))
        model.fit(X_train, y_train,validation_data=(
                        X_test, y_test), epochs=3, batch_size=64)
        return model
        
