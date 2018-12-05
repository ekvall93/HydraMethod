from modelArchitecture.ResnetRnnDense import ResnetRnnDense
from keras.losses import mean_absolute_error
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from utils import split_data, translate
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing

def compile_model():
    x = Input(shape=(32,), name='forward_pre_input')
    ResNetRnnDendeModel = ResnetRnnDense()
    y = ResNetRnnDendeModel.compile(x)
    return Model(inputs=[x], outputs=y)

if __name__ == '__main__':
    PATH = "./data/"
    splitter = split_data(PATH + "raw.csv", PATH)
    X_train, y_train, X_test, y_test = splitter.split()


    translater = translate()
    X_train = translater.translate(X_train)
    X_test = translater.translate(X_test)

    X_train = pad_sequences(X_train, maxlen=32, padding="pre")
    X_test = pad_sequences(X_test, maxlen=32, padding="pre")


    std_scale = preprocessing.StandardScaler().fit(y_train)
    y_train_std = std_scale.transform(y_train)
    y_test_std  = std_scale.transform(y_test)


    hydra = compile_model()
    hydra.compile(loss = mean_absolute_error,
                  optimizer = Adam(1e-4))
    hydra.fit(X_train, y_train_std,validation_data=(
                    X_test, y_test_std), epochs=10, batch_size=64)
