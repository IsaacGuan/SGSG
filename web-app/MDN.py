import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.models import Model

class MDN:
    def __init__(self, labels_dim=25, z_dim=128, cat_num=5):
        self.labels_dim = labels_dim
        self.z_dim = z_dim
        self.cat_num = cat_num

        self.labels_shape = (1, labels_dim)
        self.z_shape = (1, z_dim)

    def elu_modified(self, x):
        return tf.nn.elu(x) + 1

    def get_model(self):
        l_labels = Input(shape=self.labels_shape)
        l_z = Input(shape=self.z_shape)

        l_fc1 = Dense(
            units = 512,
            activation = 'tanh')(l_labels)
        l_fc2 = Dense(
            units = 1024,
            activation = 'tanh')(l_fc1)
        l_fc3 = Dense(
            units = 512,
            activation = 'tanh')(l_fc2)

        l_alpha = Dense(
            units = self.cat_num,
            activation = 'softmax')(l_fc3)
        l_mu = Dense(
            units = self.cat_num*self.z_dim)(l_fc3)
        l_sigma = Dense(
            units = self.cat_num*self.z_dim,
            activation = self.elu_modified)(l_fc3)

        return {'labels': l_labels,
                'z': l_z,
                'alpha': l_alpha,
                'mu': l_mu,
                'sigma': l_sigma}

    def test(self, weights_dir, batch_labels):
        model = self.get_model()

        model_test = Model(model['labels'], [model['alpha'], model['mu'], model['sigma']])
        model_test.load_weights(weights_dir)

        predictions = model_test.predict(batch_labels)
        tf.keras.backend.clear_session()

        predictions_alpha = predictions[0]
        predictions_mu = predictions[1]
        predictions_sigma = predictions[2]

        predictions_mu = np.reshape(predictions_mu, (np.shape(predictions_mu)[0], self.z_dim, self.cat_num))
        predictions_sigma = np.reshape(predictions_sigma, (np.shape(predictions_mu)[0], self.z_dim, self.cat_num))

        return predictions_alpha, predictions_mu, predictions_sigma
