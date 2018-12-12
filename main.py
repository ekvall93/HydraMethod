"""Markus Ekvall: 2018-12-05."""
from modelArchitecture.ResnetRnnDense import ResnetRnnDense

from keras.models import model_from_yaml
from utils import split_data, translate, save_model, standardize
from HydraMethod import HydraMethod


if __name__ == '__main__':
    PATH = "./data/"
    splitter = split_data(PATH + "raw.csv", PATH, test_size=0.05,
                          train_size=0.05)
    x_train, y_train, x_test, y_test = splitter.split()
    translater = translate()

    x_train = translater.translate(x_train)
    x_test = translater.translate(x_test)

    y_train_std, y_test_std = standardize(y_train, y_test)

    batch_size = 256

    hydra = HydraMethod(ResnetRnnDense(virtual_batch_size=int(batch_size / 8)),
                        4)
    hydra.compile()
    trained_model = hydra.fit(x_train, y_train_std, x_test, y_test_std,
                              batch_size=batch_size)
    save_model(trained_model, "./models/", "one_head_hydra")
    print("-----Finnised-----")
