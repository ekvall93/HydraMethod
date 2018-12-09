import keras
import tensorflow as tf
import keras.backend as K


import keras
import tensorflow as tf
import keras.backend as K


class VirtualBatchNormalization(keras.layers.BatchNormalization):
    """
    Identical to keras.layers.BatchNormalization, but adds the option to freeze
    parameters.
    """
    def __init__(self, virtual_batch_size=16, *args, **kwargs):
        self.virtual_batch_size = virtual_batch_size

        super(VirtualBatchNormalization, self).__init__(*args, **kwargs)

    def normalize_batch_in_training(self, x, gamma, beta,
                                    reduction_axes, epsilon=1e-3):
        return self._regular_normalize_batch_in_training(x, gamma, beta,
                                                         reduction_axes,
                                                         epsilon=epsilon)

    def _regular_normalize_batch_in_training(self, x, gamma, beta,
                                             reduction_axes, epsilon=1e-3):
        idx = tf.range(self.virtual_batch_size)
        rinput = tf.gather(x, idx)
        mean, var = tf.nn.moments(rinput, reduction_axes,
                                  None, None, False)
        normed = tf.nn.batch_normalization(x, mean, var,
                                           beta, gamma,
                                           epsilon)
        return normed, mean, var

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    axis=self.axis,
                    epsilon=self.epsilon)
            else:
                return K.batch_normalization(
                    inputs,
                    self.moving_mean,
                    self.moving_variance,
                    self.beta,
                    self.gamma,
                    axis=self.axis,
                    epsilon=self.epsilon)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference()

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = self.normalize_batch_in_training(
            inputs, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = super(VirtualBatchNormalization, self).get_config()
        config.update({'virtual_batch_size': self.virtual_batch_size})
        return config
