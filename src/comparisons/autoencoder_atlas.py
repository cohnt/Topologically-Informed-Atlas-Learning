# From https://github.com/ekorman/Atlas
# Written by Eric Korman (https://web.ma.utexas.edu/users/ekorman/)

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics

from sklearn.preprocessing import MinMaxScaler

sigma = 1.

class Atlas(object):
    '''
    d = dimension of the charts (i.e. intrinsic dimension of the manifold)
    k = number of charts
    '''
    def __init__(self, d, k):
        self.d = d
        self.k = k
        self.sc = MinMaxScaler(feature_range=(-1,1))

    def fit(self, X, val_ratio = .2, intermediate_dim=64, batch_size=100, epochs=200):
        sigma = 1.
        N = X.shape[1] # ambient dimension

        x = Input(shape=(N,))
        h = Dense(intermediate_dim, activation='relu', name='encoder_hidden')(x)
        charts = [Dense(self.d, activation='tanh',
                        name = 'chart' + str(i))(h) for i in range(self.k)]
        p = Dense(self.k, activation='softmax', name='prob_charts')(h)

        chart_hidden_lyers = [Dense(intermediate_dim, activation='relu',
                                    name='chart' + str(i) + '_hidden') for i in range(self.k)]

        chart_images_lyrs = [Dense(N, activation='tanh',
                                   name='chart' + str(i) + '_image') for i in range(self.k)]

        charts_hidden = [chart_hidden_lyers[i](charts[i]) for i in range(self.k)]

        charts_images = [chart_images_lyrs[i](charts_hidden[i]) for i in range(self.k)]

        k = self.k
        class CustomLossLayer(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(CustomLossLayer, self).__init__(**kwargs)

            def ae_loss(self, p, x, charts_images):
                rec_loss = keras.layers.add([p[:,i]/(2*sigma**2)*metrics.mean_squared_error(x, charts_images[i]) for i in range(k)])

                return K.mean(rec_loss) #- K.var(p)

            def call(self, inputs):
                p = inputs[0]
                x = inputs[1]
                charts_images = inputs[2:]

                loss = self.ae_loss(p, x, charts_images)
                self.add_loss(loss, inputs=inputs)
                # We won't actually use the output.
                return x

        y = CustomLossLayer(name='loss')([p, x, *charts_images])
        self.ae = Model(x, y)

        # scale points to between -1 and 1
        stdpts = self.sc.fit_transform(X)
        s = X.shape[0]
        val_index = np.random.randint(0,s,int(val_ratio*s))
        train_index = np.delete(np.arange(s), val_index)

        x_train, x_test = stdpts[train_index], stdpts[val_index]
        self.ae.compile(optimizer='rmsprop', loss=None)
        self.ae.fit(x_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test,None))

        # create the generators (i.e. maps from the charts to the manifold)
        decoder_inputs = [Input(shape=(self.d,)) for i in range(self.k)]
        _h_decoded = [chart_hidden_lyers[i](decoder_inputs[i]) for i in range(self.k)]
        _chart_images = [chart_images_lyrs[i](_h_decoded[i]) for i in range(self.k)]
        self.generators = [Model(decoder_inputs[i], _chart_images[i]) for i in range(self.k)]

        # create the encoder
        self.encoder = Model(x, charts + [p])

    # gives the probability of which chart a point belongs in
    def chart_probs(self, X):
        return self.encoder.predict(self.sc.transform(X))[-1]

    '''
    Maps points from chart i to the ambient space
    '''
    def decode(self, i, points):
        return self.sc.inverse_transform(self.generators[i].predict(points))

    # map some points into latent space
    def encode():
        pass
