import numpy as np
from keras import layers, Model

class rnn_model():
    def __init__(self, num_neurons, num_densor):
        joint_input = layers.Input(shape=(16, 36))
        X = layers.BatchNormalization()(joint_input)
        X = layers.Bidirectional(layers.LSTM(num_neurons))(X)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(num_densor, activation='relu')(X)
        X = layers.Dense(num_densor, activation='relu')(X)
        X = layers.Dense(11)(X)
        Y = layers.Activation('softmax')(X)
        self.model = Model(inputs=joint_input, outputs=Y)

    def train(self, X, Y, epochs, batch_size):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size)

    def test(self, X, Y):
        self.model.evaluate(X, Y)


if __name__ == '__main__':
    X = np.load('motion_fill_interpolate.npy')
    Y = np.load('label.npy')
    X_train = X[:25000, :, :]
    Y_train = Y[:25000, :]
    X_dev = X[25000:30000, :, :]
    Y_dev = Y[25000:30000, :]
    X_test = X[30000:35000, :, :]
    Y_test = Y[30000:35000, :]
    model = rnn_model(30, 11)
    model.train(X_train, Y_train, 70, 128)
    model.test(X_dev, Y_dev)
    model.test(X_test, Y_test)
