from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
#import pylab as plt
import matplotlib.pyplot as plt

def load_data(filename):
    data = np.load(filename)
    #print (data[:100, :10, :, :].shape)
    X = data[:100, :10, :, :].reshape(100, 10, 64, 64, 1)
    Y = data[:100, 10:20, :, :].reshape(100, 10, 64, 64, 1)
    return (X, Y)

if __name__ == "__main__":
    model = keras.Sequential(
        [
            keras.Input(
                shape=(10, 64, 64, 1)
            ),
            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.Conv2D(
                filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same"
            ),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adadelta")
    model.summary()
    X, Y = load_data("../train1000.npy")
    print(X.shape, Y.shape)
    model.fit(X, Y, batch_size=10, epochs=1, verbose=2, validation_split=0.1)

    # predict
    result = model.predict(X[:1, :, :, :]).reshape(10, 64, 64)
    print(result.shape)
    for i in range(10):
        plt.imshow(result[0], cmap="gray")
        plt.show()
