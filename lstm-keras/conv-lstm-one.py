from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
#import pylab as plt
import matplotlib.pyplot as plt

def load_data(filename):
    data = np.load(filename)
    #print (data[:100, :10, :, :].shape)
    print(data.shape)
    X = data[:, :10, :, :].reshape(1000, 10, 64, 64, 1)
    Y = data[:, 10:20, :, :].reshape(1000, 10, 64, 64, 1)
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
    model.fit(X[:200], Y[:200], batch_size=10, epochs=5, verbose=2, validation_split=0.1)

    # predict
    pred = model.predict(X[300:301, :, :, :]).reshape(10, 64, 64)
    actual = Y[300:301, :, :, :].reshape(10, 64, 64)

    fig, ax = plt.subplots(2, 10)
    i = 0
    for row in ax:
        for col in range(len(row)):
            if i == 1:
                row[col].imshow(pred[len(row) - col - 1], cmap="gray")
            else:
                row[col].imshow(actual[col], cmap="gray")
        i += 1
        
    plt.show()
    #print(result.shape)
    '''
    fig, ax = plt.subplots(2, 10)
    for row in ax:
        for col in range(len(row)):
            row[col].imshow(X[15][col], cmap="gray")
    plt.show()
    '''
