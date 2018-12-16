from modelArchitecture.ResnetRnnDense import ResnetRnnDense
from modelArchitecture.VirtualBatchNormalization import VirtualBatchNormalization
from modelArchitecture.Attention import Attention
from keras.models import model_from_yaml, model_from_json
from utils import split_data, translate, save_model, standardize, result
from HydraMethod import HydraMethod
import keras
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras.losses import mean_absolute_error
import math
import numpy as np
from keras.backend import eval

PATH = "./data/"
splitter = split_data(PATH + "raw.csv", PATH, train_size=0.99)
x_train, y_train= splitter.split()
translater = translate()

x_train = translater.translate(x_train)


y_train_std = standardize(y_train)

batch_size = 128
hydra = HydraMethod(ResnetRnnDense(blocks= [1, 1, 1, 1], features=32,
                                   ResnetDrop=0.1, RnnDrop=0.5,DenseDrop=0.1), 4)


x_train = hydra.get_all_representation(x_train)


load_model = False


class LR():
    def __init__(self, lr):
        self.lr = lr

    def step_decay(self, epoch):
        initial_lrate = self.lr
        drop = 0.8
        epochs_drop = 5.0
        lrate = initial_lrate * math.pow(drop,
                                         math.floor((1+epoch)/epochs_drop))
        print(lrate)
        return lrate


def freeze_it(model, block_nr):
    counter = 0
    for layer in model.layers:
        if "add" in layer.name:
            counter += 1
        if counter >= 4 * block_nr:
            layer.trainable = True
        else:
            layer.trainable = False
    print(counter / 4)
    return model


def T95(model, x, y):
    predict_train = model.predict(x)
    res_train = result(y, predict_train)
    print(np.round(res_train.t95()[0], 4))


EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              min_delta=0, patience=5,
                                              verbose=1,
                                              restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5)

def training(model, X_train, y_train_std, lr, bz, epoch, k=None):
    if k:
        print("not none")
        model = freeze_it(model, k)

    lr = LR(lr)
    lrate = LearningRateScheduler(lr.step_decay)

    model.compile(loss=mean_absolute_error, optimizer=Adam(2e-3))
    history = model.fit(X_train, y_train_std,
                        epochs=epoch, batch_size=bz,
                        callbacks=[lrate])

    T95(model, X_train, y_train_std)
    return model


if load_model:
    yaml_file = open('models/LRF_model.json', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()

    model = model_from_json(loaded_model_yaml,
                            custom_objects={'VirtualBatchNormalization':
                                            VirtualBatchNormalization,
                                            'Attention': Attention})
    model.load_weights('models/LRF_model.h5')
else:
    hydra.compile(optimizer=Adam())
    model = hydra.get_model()


lr = 2e-3
model = training(model, x_train, y_train_std, lr,
                 batch_size, 42)


# serialize model to YAML
model_yaml = model.to_yaml()
with open("models/all_data.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("models/all_data.h5")
print("Saved model to disk")