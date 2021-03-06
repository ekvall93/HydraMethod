"""Markus Ekvall: 2018-12-05."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
from keras.callbacks import Callback, LambdaCallback
import math
from matplotlib import pyplot as plt
import keras.backend as K


def check_valid(file):
    """
    Check if file exist.

    :param file: File path.
    :returns: error or filepath.
    """
    if not os.path.exists(file):
        error("The file %s does not exist!" % file)
    else:
        return(file)


def save_model(model, path, name):
    """
    Save model to yaml.

    :param model: model.
    :param path: save model path.
    :param name: save model as "name".
    """
    model_yaml = model.to_yaml()
    with open(path+name+".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
        # serialize weights to HDF5
    model.save_weights(path+name+".yaml"".h5")
    print("Saved model to disk")


def standardize(y_tr, y_te=None):
    """
    Standardize labels.

    :param y_tr: Train labels.
    :param y_te: Test labels.
    :returns: Standardized labels.
    """
    std_scale = preprocessing.StandardScaler().fit(y_tr)
    if y_te is not None:
        return std_scale.transform(y_tr), std_scale.transform(y_te)
    else:
        return std_scale.transform(y_tr)


def t95_metric(y_true, y_pred):
    """
    t95 for tensorflow.

    :param y_true: Empeircal labels
    :param y_pred: Predicted lables.
    :returns: t95
    """
    y_true, y_pred = tf.squeeze(y_true), tf.squeeze(y_pred)
    idx = tf.cast(tf.size(y_true), tf.float32)
    t95_ix = tf.cast(tf.round(tf.scalar_mul(0.95, idx)), tf.int32)
    sort_abs_res = sort(tf.squeeze(tf.abs(y_true-y_pred)))
    t95 = tf.gather(sort_abs_res, t95_ix)
    numerator = tf.scalar_mul(2, t95)
    denominator = tf.reduce_max(y_true) - tf.reduce_min(y_true)
    T = tf.div(numerator, denominator)
    return tf.scalar_mul(100, T)


class split_data():
    """Split train/test-set. No common sequence between sets."""
    def __init__(self, in_path, out_path, prepare_csv=False, test_size=None,
                 train_size=1.0, random_state=42):
        """
        :param in_path: Path of input files.
        :param out_path: Path for output files.
        :param prepare_csv: Save sets in csv files.
        :param test_size: Size of test set.
        :param train_size: Size of train set.
        :param random_state: Random state.

        :returns: Train and test sets.
        """
        check_valid(in_path)
        if test_size:
            assert(test_size + train_size <= 1.0,
                "The combined splits can not exceed 100%")
        else:
            assert(train_size <= 1.0,
                "The combined splits can not exceed 100%")

        self.in_path = in_path
        self.out_path = out_path
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def get_ix(self, n):
        """
        Get shuffled indices.

        :param n: Number of indices.
        :returns: Shuffled indices.
        """
        ix = np.arange(n)
        np.random.shuffle(ix)
        return ix

    def split_ix(self, X, idx, train=True):
        """
        Split indices.

        :param X: Unique sequences.
        :param idx: Unique idx.
        :returns: Split unique sequences.
        """
        train_ix, test_ix = train_test_split(idx, test_size=self.test_size,
                                             train_size=self.train_size,
                                             random_state=self.random_state)
        return X[train_ix].tolist(), X[test_ix].tolist()

    def get_datasets(self, X, idx):
        """
        Get det dataset.

        :param X: THe whole dataset.
        :param idx: Unique seqeunces.
        :returns: Dataset with no common sequence as the other set.
        """
        return X.copy()[X['mod_sequence'].isin(idx)]

    def save_csv(self, X, name):
        """
        Save dataset.

        :param X: Dataset.
        :param name: Name of file for dataset.
        :returns: True
        """
        X['mod_sequence'].to_csv(self.out_path + "seq_" + name, index=False)
        X['irt'].to_csv(self.out_path + "irt_" + name, index=False)
        return True

    def split(self):
        """
        Split dataset into test and train.

        :returns: Train and test dataset.
        """

        df = pd.read_csv(self.in_path)
        new_df = df[['irt', 'mod_sequence']].copy()
        new_df["mod_sequence"] = new_df["mod_sequence"].map(
            lambda x: x.replace("-CaC-", "C").replace("-OxM-", "O"))
        unique_seq = new_df['mod_sequence'].unique()

        ix = self.get_ix(len(unique_seq))

        if self.test_size:


            train_ix, test_ix = self.split_ix(unique_seq, ix)

            train_set = self.get_datasets(new_df, train_ix)
            test_set = self.get_datasets(new_df, test_ix)

            self.save_csv(train_set, "train")
            self.save_csv(test_set, "test")

            return (train_set['mod_sequence'].tolist(),
                    train_set['irt'].values.reshape((-1, 1)),
                    test_set['mod_sequence'].tolist(),
                    test_set['irt'].values.reshape((-1, 1)))
        else:

            #train_ix = ix[:int(len(ix)*self.train_size)]
            train_ix = ix
            train_ix = unique_seq[train_ix].tolist()
            print(len(train_ix))
            train_set = self.get_datasets(new_df, train_ix)
            self.save_csv(train_set, "train")

            return (train_set['mod_sequence'].tolist(),
                    train_set['irt'].values.reshape((-1, 1)))



class translate():
    """Translate sequence of letters to numbers."""
    def get_dict(self):
        """
        Split dataset into test and train.

        :returns: Peptide dict.
        """
        peptide_dict = {'L': 1, 'S': 2, 'E': 3, 'A': 4, 'G': 5,
                        'P': 6, 'V': 7, 'K': 8, 'T': 9, 'D': 10,
                        'R': 11, 'Q': 12, 'I': 13, 'F': 14, 'N': 15,
                        'Y': 16, 'H': 17, 'O': 18, 'M': 19, 'W': 20, 'C': 21}
        return peptide_dict

    def translate(self, seq_data):
        """
        Translate all sequences in dataset.

        :param seq_data: Sequence dataset.
        :returns: Translated dataset.
        """
        peptide_dict = self.get_dict()
        return self.make_translation(peptide_dict, seq_data)

    def make_translation(self, peptide_dict, seq_data):
        """
        Translate sequence.

        :param peptide_dict: Peptide alphabet.
        :param seq_data: Sequence to translate.
        :returns: Translated sequence.
        """
        for i, seq in enumerate(seq_data):
            seq_data[i] = [peptide_dict[s] for s in seq]
        return seq_data

    def reverse_translation(self, num_seq_data):
        """
        Translate numbers to letters.

        :param num_seq_data: Sequence data based on number.
        :returns: Sequence based on letters.
        """
        peptide_dict = self.get_dict()
        peptide_dict_inv = {v: k for k, v in peptide_dict.items()}
        rev_pep_sec = []
        for pep_sec in num_seq_data:
            sec = [peptide_dict_inv[a] for a in pep_sec if a != 0]
            " ".join(sec)
            rev_pep_sec += [sec]
        return rev_pep_sec


class data_split():
    """Split train/test-set. Common sequence between sets."""
    def __init__(self, validation_split, train_size=0.90, test_split=None,
                 random_state=42):
        """
        :param validation_split: validation set.
        :param train_size: Train size.
        :param test_split: Test size.
        :param random_state: Random state.

        :returns: Train, validation and test sets.
        """
        assert(validation_split + test_split + train_size <= 1.0,
               "The combined splits can not exceed 100%")
        self.validation_split = validation_split
        self.test_split = test_split
        if test_split is not None:
            self.split = validation_split + test_split
        else:
            self.split = validation_split
        self.random_state = random_state
        self.train_size = train_size

    def get_data(self, X, y):
        """
        Split data.

        :param X: input data.
        :param y: targtes.
        :returns: Train, test, and validation set.
        """
        X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=self.split, train_size=self.train_size,
                        random_state=self.random_state)
        if self.test_split is not None:
            _, test_ratio = self.get_ratio()
            X_val, X_test, y_val, y_test = train_test_split(
                   X_val, y_val, test_size=test_ratio,
                   random_state=self.random_state)
            return X_train, X_val, X_test, y_train, y_val, y_test

        return X_train, X_val, y_train, y_val

    def get_ratio(self):
        """
        Get the ratio between test and train set.

        :returns: Test/valid - ratio.
        """
        test_ratio = 1 / ((self.validation_split / self.test_split) + 1)
        val_ratio = self.validation_split / (self.validation_split +
                                             self.test_split)
        return val_ratio, test_ratio


class result():
    """Different result metrics i.e., t95 and MSE."""
    def __init__(self, emp, preds):
        """
        Split data.

        :param emp: Empeircal labels
        :param preds: Predicted lables.
        """
        self.emperical = emp
        self.model = preds
        self.t_95 = int(emp.shape[0] * 0.95)

    def rmse(self):
        """
        Root Mean Square

        :returns: rmse
        """
        return mse(self.emperical, self.model)**0.5

    def t95(self):
        """
        t95

        :returns: t95
        """
        error = self.abs_dif_sort()
        return 2 * error[self.t_95] / (max(self.emperical) -
                                       min(self.emperical)) * 100

    def abs_dif_sort(self):
        """
        Absolute difference of emperical and predicted labels.

        :returns: t95
        """
        return np.sort(abs(self.emperical-self.model).flatten())


class TestCallback(Callback):
    """t95 for keras Callback."""
    def __init__(self, emp, preds):
        """
        :param emp: Empeircal labels
        :param preds: Predicted lables.
        """
    def on_train_begin(self, logs={}):
        """
        set t95

        :param logs: log dict.
        """
        self.t95 = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Get t95

        :param epoch: epoch.
        :param logs: log dict.
        :returns: t95
        """
        val_predict = self.model.predict(self.validation_data[0])
        val_targ = self.validation_data[1]
        _t95 = self.calc_t95(val_targ, val_predict)
        self.t95.append(_t95[0])
        print(" — t95: %f" % (_t95))
        return _t95

    def calc_t95(self, y_true, y_pred):
        """
        Calculate t95

        :param y_true: True labels.
        :param y_pred: predicted labels.

        :returns: t95
        """
        sample_size = y_true.shape[0]
        sample_95 = int(sample_size * 0.95)
        error = y_true - y_pred
        t = np.sort(np.abs(error.flatten()))[sample_95]
        t95 = 2 * t / (max(y_true) - min(y_true)) * 100
        return t95


class LRFinder:
    """
    Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
    See for details:
    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    """
    def __init__(self, model):
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        # Log the learning rate
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # Log the loss
        loss = logs['loss']
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        if math.isnan(loss) or loss > self.best_loss * 4:
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, x_train, y_train, start_lr, end_lr, batch_size=64,
             epochs=1, fit=True):
        num_batches = epochs * y_train.shape[0] / batch_size
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) /
                                                             float(num_batches)
                                                             )

        # Save weights into a file
        self.model.save_weights('tmp.h5')

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)
        callback = LambdaCallback(on_batch_end=lambda batch,
                                  logs: self.on_batch_end(batch, logs))
        if fit:
            self.model.fit(x_train, y_train, batch_size=batch_size,
                           epochs=epochs, callbacks=[callback])
        else:
            self.model.fit_generator(generator=x_train, epochs=epochs,
                                     steps_per_epoch=(y_train.shape[0] /
                                                      batch_size),
                                     callbacks=[callback])

        #Restore the weights to the state before model fitting
        #self.model.load_weights('tmp.h5')
        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    def plot_loss(self, n_skip_beginning=10, n_skip_end=5):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end],
                 self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale('log')

    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5,
                         y_lim=(-0.01, 0.01)):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivative = (self.losses[i] - self.losses[i - sma]) / sma
            derivatives.append(derivative)

        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end],
                 derivatives[n_skip_beginning:-n_skip_end])
        plt.xscale('log')
        plt.ylim(y_lim)
