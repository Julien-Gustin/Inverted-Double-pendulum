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

        self.add(keras.layers.Dense(output, kernel_initializer=keras.initializers.RandomUniform(-0.5, 0.5), bias_initializer=keras.initializers.RandomUniform(-0.5, 0.5)))
        self.compile(loss="mse", optimizer="adam")
        self.epochs = epochs
        self.batch_size = batch_size


    def fit(self, *args, **kwargs):
        kwargs.setdefault('batch_size', self.batch_size)
        kwargs.setdefault('epochs', self.epochs)
        kwargs.setdefault('verbose', False)
        super().fit(*args, **kwargs)
