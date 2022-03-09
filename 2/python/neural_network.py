from tabnanny import verbose
import tensorflow.keras as keras

class NN(keras.Sequential):
    """ Neural Network """

    # https://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf
    def __init__(self, layers, neurons, output, epochs, batch_size ,activation="sigmoid", name="Neural network"):
        super().__init__()
        self.neurons = neurons
        self.activation = activation
        self.n_layers = layers

        self.add(keras.layers.Flatten())

        for _ in range(layers):
            self.add(keras.layers.Dense(neurons, activation=self.activation, kernel_initializer=keras.initializers.RandomUniform(-0.5, 0.5), bias_initializer=keras.initializers.RandomUniform(-0.5, 0.5)))

        self.add(keras.layers.Dense(output, activation, kernel_initializer=keras.initializers.RandomUniform(-0.5, 0.5), bias_initializer=keras.initializers.RandomUniform(-0.5, 0.5)))
        self.compile(loss="mse", optimizer="adam")
        self.epochs = epochs
        self.batch_size = batch_size


    def fit(self, X, y):
        self.__init__(self.n_layers, self.neurons, 1, self.epochs, self.batch_size)
        super().fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=False)
