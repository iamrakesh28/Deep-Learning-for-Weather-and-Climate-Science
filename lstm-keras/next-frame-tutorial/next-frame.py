from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt

# Artificial data generation

# Generate movies with 3 to 7 moving squares inside.
# The squares are of shape 1x1 or 2x2 pixels, which move linearly over time.
# For convenience we first create movies with bigger width and height (80x80)
# and at the end we select a 40x40 window.

def generate_movies(n_samples=1200, n_frames=20):
    row = 80
    col = 80
    orginal_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)

    for i in range(n_samples):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)

        for j in range(n):
            # Initial position
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # Direction of motion
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1

            # Size of the square
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                orginal_movies[
                    i, t, x_shift - w : x_shift + w, y_shift - w : y_shift + w, 0
                ] += 1

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[
                    i, t, x_shift - w : x_shift + w, y_shift - w : y_shift + w, 0
                ] += 1

    # Cut to a 40x40 window
    orginal_movies = orginal_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    orginal_movies[orginal_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return orginal_movies, shifted_movies


# Testing the network on one movie
# Feed it with the first 10 positions and then predict the new positions

def prediction(seq, which, orginal_movies, shifted_movies):
    track = orginal_movies[which][:10, ::, ::, ::]

    # track has the shape of 10 frames 40*40 with one channel. np.newaxis
    # adds additional axis so the array can be accepted by the seq model

    for j in range(10):
        new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::]) # (1, 10, 40, 40, 1)
        new = new_pos[::, -1, ::, ::, ::] # (1, 40, 40, 1)
        # adds +1 to the first dimension in each loop cycle
        track = np.concatenate((track, new), axis=0)

    # compare the predictions to the ground truth
    
    track2 = orginal_movies[which][10:, ::, ::, ::]
    for i in range(20):
        if i < 10:
            plt.title("Actucal_" + str(i + 1))
            plt.imshow(track[i, :, :, 0])
            plt.show()
            continue
        plt.subplot(121), plt.imshow(track[i, :, :, 0]), plt.title("Predicted_" + str(i + 1))
        plt.subplot(122), plt.imshow(track2[i - 10, :, :, 0]), plt.title("Actual_" + str(i + 1))
        plt.show()

    
if __name__ == "__main__":

    # Model
    # We create a layer which take as input movies of shape
    # (n_frames, width, height, channels) and returns a movie of identical shape
    
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       input_shape=(None, 40, 40, 1),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    
    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation='sigmoid',
               padding='same', data_format='channels_last'))
    seq.compile(loss='binary_crossentropy', optimizer='adadelta')

    seq.summary()

    orginal_movies, shifted_movies = generate_movies(n_samples=1200)

    # Train the network
    seq.fit(orginal_movies[:1000], shifted_movies[:1000],
        epochs=50, validation_split=0.1)

    # prediction
    prediction(seq, 1004, orginal_movies, shifted_movies)

