import pandas as pd
import numpy as np


class split_data():
    def __init__(self, in_path, out_path, prepare_csv=False,test_split =0.0,
                 train_size=1.0, *args, **kwargs):
        is_valid_file(in_path)
        assert test_split + train_size <= 1.0,"The combined splits can not exceed 100%"
        self.in_path = in_path
        self.out_path = out_path
        super(split_data, self).__init__(*args, **kwargs)

    def get_ix(self,n):
        ix = np.arange(n);np.random.shuffle(ix)
        return ix

    def split_ix(self, X, idx, test_split=0.1, train=True):
        n = int(len(idx)*test_split)
        if train:
            return X[idx[n:]].tolist()
        else:
            return X[idx[:n]].tolist()

    def get_datasets(self, X, idx):
        return X.copy()[X['mod_sequence'].isin(idx)]

    def save_csv(self, X, name):
        X['mod_sequence'].to_csv(self.out_path + "seq_" + name,index=False)
        X['irt'].to_csv(self.out_path + "irt_" + name,index=False)
        return True

    def split(self, verbose=True, return_data=True):

        df = pd.read_csv(self.in_path)
        new_df = df[['irt', 'mod_sequence']].copy()
        new_df["mod_sequence"] = new_df["mod_sequence"].map(lambda x: x.replace("-CaC-","C").replace("-OxM-","O"))
        unique_seq = new_df['mod_sequence'].unique()

        ix = self.get_ix(len(unique_seq))

        train_ix = self.split_ix(unique_seq, ix)
        test_ix= self.split_ix(unique_seq, ix, train=False)

        train_set = self.get_datasets(new_df, train_ix)
        test_set = self.get_datasets(new_df, test_ix)

        self.save_csv(train_set, "train")
        self.save_csv(test_set, "test")

        if return_data:
            return train_set['mod_sequence'].tolist(), train_set['irt'].values.reshape((-1,1)), test_set['mod_sequence'].tolist(), test_set['irt'].values.reshape((-1,1))
        else:
            return True



def rmse(predict, emperical): return mse(predict, emperical)**0.5

def treshold(t): return np.sum([abs(predict_test - y_test) < t])

def fun_opt(t): return np.sum([abs(predict_test - y_test) < t]) - t_s

def cut_off(): return newton(fun_opt,0.02)

def t_95(cut_off): return 2 * cut_off /(max(y_test) - min(y_test)) * 10

def fit_plot(predict_test,test_target):
    import seaborn as sns
    plt.figure()
    x,y = predict_test, test_target
    # Set up the axes with gridspec
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(4, 4, hspace=0.6, wspace=1.4)
    main_ax = fig.add_subplot(grid[:-1, 1:])
    y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)
    x_hist.hist(x, int(round(np.sqrt(np.size(x)))), histtype='stepfilled',
                orientation='vertical', color='green')
    x_hist.invert_yaxis()
    minimum = min(min(x),min(y))
    maximum = max(max(x),max(y))
    main_ax.plot([minimum,maximum],[minimum,maximum],'k-')
    main_ax.set_xlabel("Predicted retentiontimes [s]")
    main_ax.set_ylabel("Emperical retentiontimes [s]")
    plt.show()

import time
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        time_var = float((time2-time1)*1000.0)
        if time_var > float(60000.0):
            sec = (time_var % float(60000.0))/1000.0
            min = int(time_var / float(60000.0))
            print('{:s} function took {:.3f} min and {:.3f} s'.format(f.__name__, min , sec ))
        elif time_var > float(1000.0):
            print('{:s} function took {:.3f} s'.format(f.__name__, (time2-time1)*1.0))
        else:
            print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap


class translate():
    def get_dict(self):

        peptide_dict = {'L': 1,'S': 2,'E': 3,'A': 4,'G': 5,'P': 6,'V': 7,'K': 8,
         'T': 9,'D': 10,'R': 11,'Q': 12,'I': 13,'F': 14,'N': 15,
         'Y': 16,'H': 17,'O': 18,'M': 19,'W': 20}
        return peptide_dict
    @timing
    def translate(self, seq_data):
        peptide_dict = self.get_dict()
        return self.make_translation(peptide_dict, seq_data)

    def make_translation(self,peptide_dict, seq_data):
        for i, seq in enumerate(seq_data):
            seq_data[i] = [peptide_dict[s] for s in seq]
        return seq_data
    def reverse_translation(self,num_seq_data):
        peptide_dict = self.get_dict()
        peptide_dict_inv = {v: k for k, v in peptide_dict.items()}
        rev_pep_sec = []
        for pep_sec in num_seq_data:
            sec = [peptide_dict_inv[a] for a in pep_sec if a !=0]
            " ".join(sec)
            rev_pep_sec += [sec]
        return rev_pep_sec



class pader():
    def zerolistmaker(self, n):
        arrayofzeros = np.zeros((1,n))
        return arrayofzeros
    @timing
    def pad(self, seq_data, pep_len):
        pad_seq= np.zeros((len(seq_data),pep_len))
        for i, seq in enumerate(seq_data):
            pad_seq[i] = np.append(self.zerolistmaker(pep_len - len(seq)),seq)
        return pad_seq


class data_split():
    def __init__(self, validation_split, train_size=0.90,test_split=None, random_state=42):
        assert validation_split + test_split + train_size <= 1.0,"The combined splits can not exceed 100%"


        self.validation_split = validation_split
        self.test_split = test_split
        if test_split is not None:
            self.split = validation_split + test_split
        else:
            self.split = validation_split
        self.random_state = random_state
        self.train_size = train_size
    def get_data(self,X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.split, train_size=self.train_size, random_state=self.random_state)
        if self.test_split is not None:
            _ , test_ratio = self.get_ratio()
            X_val, X_test, y_val, y_test = train_test_split(X_val,y_val,test_size=test_ratio,random_state=self.random_state)
            return X_train, X_val,X_test, y_train, y_val, y_test

        return X_train, X_val, y_train, y_val
    def get_ratio(self):
        test_ratio = 1 /( (self.validation_split/ self.test_split) + 1)
        val_ratio = self.validation_split / (self.validation_split + self.test_split)
        return val_ratio, test_ratio


class result():
    def __init__(self,emp, preds):
        self.emperical = emp
        self.model = preds
        self.t_95 = int(emp.shape[0] * 0.95)

    def rmse(self):
        return mse(self.emperical, self.model)**0.5

    def t95(self):
        error = self.abs_dif_sort()
        return 2 * error[self.t_95] /(max(self.emperical) - min(self.emperical)) * 100

    def abs_dif_sort(self):
        return np.sort(abs(self.emperical-self.model).flatten())

def t95_metric(y_true, y_pred):
    y_true, y_pred = tf.squeeze(y_true), tf.squeeze(y_pred)
    idx = tf.cast(tf.size(y_true),tf.float32)
    t95_ix = tf.cast(tf.round(tf.scalar_mul(0.95,idx)),tf.int32)
    sort_abs_res = sort(tf.squeeze(tf.abs(y_true-y_pred)))
    t95 = tf.gather(sort_abs_res,t95_ix)
    numerator = tf.scalar_mul(2,t95)
    denominator = tf.reduce_max(y_true) - tf.reduce_min(y_true)
    T = tf.div(numerator,denominator)
    return tf.scalar_mul(100,T)

# class TestCallback(Callback):
#     def on_train_begin(self, logs={}):
#         self.t95 = []
#
#
#     def on_epoch_end(self, epoch, logs={}):
#         val_predict = self.model.predict(self.validation_data[0])
#         val_targ = self.validation_data[1]
#         #x, y = self.test_data
#         #pred = self.model.predict(x)
#         _t95 = self.calc_t95(val_targ, val_predict)
#         self.t95.append(_t95[0])
#         print(" — t95: %f" %(_t95))
#
#         #print('\nTesting loss: {}\n'.format(err))
#         return _t95
#
#     def calc_t95(self,y_true, y_pred):
#         sample_size = y_true.shape[0]
#         sample_95 = int(sample_size * 0.95)
#         error = y_true - y_pred
#         t = np.sort(np.abs(error.flatten()))[sample_95]
#         t95 = 2 * t /(max(y_true) - min(y_true)) * 100
#         return t95
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

    def find(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1):
        num_batches = epochs * x_train.shape[0] / batch_size
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))

        # Save weights into a file
        self.model.save_weights('tmp.h5')

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        self.model.fit(x_train, y_train,
                        batch_size=batch_size, epochs=epochs,
                        callbacks=[callback])

        # Restore the weights to the state before model fitting
        self.model.load_weights('tmp.h5')

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
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale('log')

    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
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
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], derivatives[n_skip_beginning:-n_skip_end])
        plt.xscale('log')
        plt.ylim(y_lim)

import os

def is_valid_file(arg) -> str:
    """
    Credit: Tobias
    """
    if not os.path.exists(arg):
        error("The file %s does not exist!" % arg)
    else:
        return(arg)

def get_fullpath_basename(file_path: str):
    """
    Credit: Tobias
    """
    """
    >>> get_fullpath_basename("/home/tobiass/tim.tof")
    ('/home/tobiass', 'tim', '/home/tobiass/tim.tof')
    """
    fullPath = os.path.abspath(file_path)
    base_name = os.path.basename(fullPath)
    dirName = os.path.dirname(fullPath)
    base_name_without_extension = os.path.splitext(base_name)[0]

    return(dirName, base_name_without_extension, fullPath)
