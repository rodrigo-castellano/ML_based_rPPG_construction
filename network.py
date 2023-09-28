import numpy as np
import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers

''' LSTM '''



class LSTM(tfk.Model):
    def __init__(self, n_samples, n_methods, n_landmarks,drop=.1,units=None):
        super(LSTM, self).__init__()
        features = n_methods * n_landmarks
        if units==None:
            self.units = np.linspace(features, 1, 3, dtype=int)
        else:
            self.units = units
        print('units',self.units)

        input_layer = tfk.Input(shape=(n_samples, features))
        x = self.create_lstm_block(input_layer, self.units, drop)
        x = tfkl.Dense(1)(x)
        
        self.model = tfk.Model(inputs=input_layer, outputs=x)

    def call(self, inputs, training=False):
        return self.model(inputs)
    
    def create_lstm_block(self, x, units, drop):
        for unit in units:
            x = tfkl.LSTM(unit, return_sequences=True, dropout=drop, recurrent_dropout=drop)(x)
        return x


''' LSTM residual ''' 

class LSTM_res(tfk.Model):
    def __init__(self, n_samples, n_methods, n_landmarks, drop=.1, units=None):
        super(LSTM_res, self).__init__()

        features = n_methods * n_landmarks
        # set default units if none is provided
        if units is None:
            self.units = np.linspace(features, 1, 3, dtype=int)
        else:
            self.units = units
        print('units', self.units)

        # define input layer
        input_layer = tfk.Input(shape=(n_samples, features))
        # create LSTM block
        x = self.create_lstm_block(input_layer, self.units, drop)
        # add a dense layer for output
        x = tfkl.Dense(1)(x)

        # create the model with input and output layers
        self.model = tfk.Model(inputs=input_layer, outputs=x)

    def call(self, inputs, training=False):
        # call the created model with input
        return self.model(inputs)

    def create_lstm_block(self, x, units, drop):
        for i in range(len(units)):
            if i == 0:
                # first layer, no residual connection
                x = tfkl.LSTM(units[i], return_sequences=True, dropout=drop, recurrent_dropout=drop)(x)
            else:
                # intermediate layers with residual connection
                residual = x
                residual = tfkl.Dense(units[i])(residual) # transform residual to match shape of x
                x = tfkl.LSTM(units[i], return_sequences=True, dropout=drop, recurrent_dropout=drop)(x)
                x = tfkl.add([x, residual])
        return x

''' LSTM time2vec '''

class Time2Vec(tfkl.Layer):
    def __init__(self, output_dim, activation='relu', **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        super(Time2Vec, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1, input_shape[-1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Time2Vec, self).build(input_shape)

    def call(self, inputs):
        time = inputs[:,:,0]
        time = tf.expand_dims(time, axis=-1)
        x = tf.math.multiply(time, self.kernel)

        if self.activation == 'sin':
            x = tf.concat([tf.math.sin(x), tf.math.cos(x)], axis=-1)
        elif self.activation == 'tanh':
            x = tf.concat([tf.math.tanh(x), tf.math.tanh(-x)], axis=-1)
        elif self.activation == 'relu':
            x = tf.nn.relu(x)
        
        return tf.concat([x, inputs[:,:,1:]], axis=-1)

    def get_config(self):
        config = super(Time2Vec, self).get_config()
        config.update({'output_dim': self.output_dim, 'activation': self.activation})
        return config


class LSTM_T2V(tf.keras.Model):
    def __init__(self, n_samples, n_methods, n_landmarks, drop=.1, n_units=3, units=None):
        super(LSTM_T2V, self).__init__()
        features = n_methods * n_landmarks
        if units == None:
            self.units = np.linspace(features, 1, n_units, dtype=int)
        else:
            self.units = units

        print('units', self.units)

        input_layer = tfk.Input(shape=(n_samples, features))
        
        # Add Time2Vec layer
        time_embedding = Time2Vec(output_dim=features)(input_layer)
        time_embedding = Time2Vec(output_dim=features)(input_layer)
        
        # Pass the time embedding to the LSTM layers
        lstm_input = tf.concat([time_embedding, input_layer], axis=-1)
        x = self.create_lstm_block(lstm_input, self.units, drop)
        x = tfkl.Dense(1)(x)
        
        self.model = tf.keras.Model(inputs=input_layer, outputs=x)

    def call(self, inputs, training=False):
        return self.model(inputs)

    def create_lstm_block(self, x, units, drop):
        for unit in units:
            x = tfkl.LSTM(unit, return_sequences=True, dropout=drop, recurrent_dropout=drop)(x)
        return x

 
class LSTM_bioutput(tfk.Model):
    def __init__(self, n_samples, n_methods, n_landmarks,drop=.1,units=None):
        super(LSTM_bioutput, self).__init__()
        features = n_methods * n_landmarks
        if units==None:
            self.units = np.linspace(features, 1, 3, dtype=int)
        else:
            self.units = units

        input_layer = tfk.Input(shape=(n_samples, features), name='input')
        x = self.create_lstm_block(input_layer, self.units, drop)

        rPPG = tfkl.Dense(1, name='rPPG')(x)
        # # Add dropout layer, then dense layer, then flatten layer
        # bpm = tfkl.Dropout(drop)(x)
        # bpm = tfkl.Dense(1)(bpm)
        # bpm = tfkl.Flatten(name='bpm')(bpm)    
 
        bpm = tfkl.Dropout(drop)(x)
        bpm = tfkl.Dense(1)(bpm)
        bpm = tfkl.Flatten()(bpm)    
        bpm = tfkl.Dropout(drop)(bpm)
        bpm = tfkl.Dense(300)(bpm)
        bpm = tfkl.Dense(1,name='bpm')(bpm)

        self.model = tfk.Model(inputs=input_layer, outputs= {
                                                            # 'rPPG': rPPG, 
                                                            'bpm': bpm
                                                            }, 
                                                            name='LSTM_bioutput')

    def call(self, inputs, training=False):
        return self.model(inputs)
    
    def create_lstm_block(self, x, units, drop):
        for unit in units:
            x = tfkl.LSTM(unit, return_sequences=True, dropout=drop, recurrent_dropout=drop)(x)
        return x


''' CNN '''

class CNN(tfk.Model):
    def __init__(self, n_samples, n_methods, n_landmarks, drop=.1, units=None,filter_size=3,n_filters=32):
        super(CNN, self).__init__()
        features = n_methods*n_landmarks
        if units==None:
            self.units = np.linspace(features, 1, 3, dtype=int)
        else:
            self.units = units
        print('units',self.units)
        self.n_samples = n_samples
        self.n_methods = n_methods
        self.n_landmarks = n_landmarks
        self.drop = drop

        input_layer = tfk.Input(shape=(n_samples, features))
        x = tfkl.Reshape((n_samples, n_methods, n_landmarks))(input_layer)
        x = self.create_cnn_block(x, filter_size, n_filters, drop)
        x = tfkl.Dense(1)(x)
        
        self.model = tfk.Model(inputs=input_layer, outputs=x)
        # print('model summary f')
        # self.model.summary()

    def call(self, inputs, training=False):
        return self.model(inputs)
 

    def create_cnn_block(self, x, filter_size, n_filters, drop):
        # # without dropout, really overfits

        x = tfkl.Conv2D(30, (self.n_samples, 4), padding='same')(x)
        x = tfkl.BatchNormalization()(x)
        # x = tfkl.Dropout(rate=self.drop)(x)
        x = tfkl.MaxPooling2D((1, 2))(x)

        x = tfkl.Conv2D(20, (self.n_samples, 2), padding='same')(x)
        x = tfkl.BatchNormalization()(x)
        # x = tfkl.Dropout(rate=self.drop)(x)
        x = tfkl.MaxPooling2D((1, 2))(x)

        x = tfkl.Conv2D(10, (self.n_samples, 1), padding='same')(x)
        x = tfkl.BatchNormalization()(x)    
        # x = tfkl.Dropout(rate=self.drop)(x)

        x = tfkl.Conv2D(1, (self.n_samples, 1), padding='same')(x)
        x = tfkl.BatchNormalization()(x)    
        # x = tfkl.Dropout(rate=self.drop)(x)
        x = tfkl.Reshape((self.n_samples,1))(x)

        return x

class CNN1D(tfk.Model):
    def __init__(self, n_samples, n_methods, n_landmarks, drop=.1, n_units=3,units=None,filter_size=3,n_filters=32):
        super(CNN1D, self).__init__()
        features = n_methods*n_landmarks
        if units==None:
            # self.units = [features,int(features),int(features)] 
            self.units = np.linspace(features, 1, n_units, dtype=int)
        else:
            self.units = units
        print('units',self.units)
        self.n_samples = n_samples
        self.n_methods = n_methods
        self.n_landmarks = n_landmarks

        input_layer = tfk.Input(shape=(n_samples,n_methods,n_landmarks))
        x = self.create_cnn_block(input_layer, filter_size, n_filters, drop)
        x = tfkl.Dense(1)(x)
        
        self.model = tfk.Model(inputs=input_layer, outputs=x)
        print('model summary f')
        self.model.summary()

    def call(self, inputs, training=False):
        return self.model(inputs)
    
    def create_lstm_block(self, x, units, drop):
        for unit in units:
            x = tfkl.LSTM(unit, return_sequences=True, dropout=drop, recurrent_dropout=drop)(x)
        return x

    def create_cnn_block(self, x, filter_size, n_filters, drop):
        x = tfkl.Conv1D(filters=20, kernel_size=30, padding='same')(x)
        x = tfkl.Conv1D(filters=10, kernel_size=30, padding='same')(x)
        x = tfkl.Conv1D(filters=1, kernel_size=30, padding='same')(x)
        x = tfkl.Reshape((120, 4))(x)
        x = tfkl.Conv1D(filters=1, kernel_size=30, padding='same')(x)
        return x

class Dense(tfk.Model):
    def __init__(self, n_samples, n_methods, n_landmarks, drop=.1, n_units=3,units=None,filter_size=3,n_filters=32):
        super(Dense, self).__init__()
        features = n_methods*n_landmarks
        if units==None:
            self.units = np.linspace(features, 1, n_units, dtype=int)
        else:
            self.units = units
        print('units',self.units)
        self.n_samples = n_samples
        self.n_methods = n_methods
        self.n_landmarks = n_landmarks

        input_layer = tfk.Input(shape=(n_samples,n_methods,n_landmarks))
        x = self.create_dense_block(input_layer, filter_size, n_filters, drop)
        x = Dense(1)(x)
        
        self.model = tfk.Model(inputs=input_layer, outputs=x)
        print('model summary f')
        self.model.summary()

    def call(self, inputs, training=False):
        return self.model(inputs)
    
    def create_lstm_block(self, x, units, drop):
        for unit in units:
            x = tfkl.LSTM(unit, return_sequences=True, dropout=drop, recurrent_dropout=drop)(x)
        return x

    def create_dense_block(self, x, filter_size, n_filters, drop):
        # # 0.03
        x = tfkl.Reshape((self.n_samples, self.n_methods*self.n_landmarks))(x)
        x = tfkl.Dense(120)(x)
        x = tfkl.Dense(60)(x)
        x = tfkl.Dense(30)(x)
        x = tfkl.Dropout(drop)(x)
        return x