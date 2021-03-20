import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K
from tensorflow_probability import distributions as tfd

class MDN:
    def __init__(self, labels_dim, z_dim=128, cat_num=5):
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

    def mdn_loss(self, z, alpha, mu, sigma):
        alpha = K.repeat_elements(alpha, self.z_dim, axis=1)
        alpha = K.expand_dims(alpha, axis=3)

        mu = K.reshape(mu, (tf.shape(mu)[0], self.z_dim, self.cat_num))
        mu = K.expand_dims(mu, axis=3)

        sigma = K.reshape(sigma, (tf.shape(sigma)[0], self.z_dim, self.cat_num))
        sigma = K.expand_dims(sigma, axis=3)

        gm = tfd.MixtureSameFamily(
            mixture_distribution = tfd.Categorical(probs=alpha),
            components_distribution = tfd.Normal(
                loc = mu,
                scale = sigma))

        z = tf.transpose(z, (0, 2, 1))

        return tf.reduce_mean(-gm.log_prob(z))

    def train(self, learning_rate, batch_size, epoch_num, labels_train, z_train):
        model = self.get_model()

        labels = model['labels']
        z = model['z']
        alpha = model['alpha']
        mu = model['mu']
        sigma = model['sigma']

        model_train = Model([labels, z], [alpha, mu, sigma])

        model_train.add_loss(self.mdn_loss(z, alpha, mu, sigma))

        adam = Adam(lr=learning_rate)
        model_train.compile(optimizer=adam)

        model_train.fit(
            [labels_train, z_train],
            batch_size = batch_size,
            epochs = epoch_num,
            validation_data = ([labels_train, z_train], None)
        )

        return model_train

    def test(self, labels_test, weights_dir):
        model = self.get_model()

        model_test = Model(model['labels'], [model['alpha'], model['mu'], model['sigma']])
        model_test.load_weights(weights_dir)

        predictions = model_test.predict(labels_test)

        predictions_alpha = predictions[0]
        predictions_mu = predictions[1]
        predictions_sigma = predictions[2]

        predictions_mu = np.reshape(predictions_mu, (np.shape(predictions_mu)[0], self.z_dim, self.cat_num))
        predictions_sigma = np.reshape(predictions_sigma, (np.shape(predictions_mu)[0], self.z_dim, self.cat_num))

        return predictions_alpha, predictions_mu, predictions_sigma
