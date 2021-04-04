import tensorflow as tf
import numpy as np
from encoder_decoder import EncoderDecoder
from test import test_model
    
def load_dataset(path, filename):
    train_data = np.load(path + filename)
    # patch size 4 x 4
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 50, 50, 4)

    train_data[train_data < 200] = 0
    train_data[train_data >= 200] = 1
    #train_data = train_data / 255.0
    print(train_data.shape)
    
    X = train_data[:, :10, :, :, :]
    Y = train_data[:, 10:21, :, :, :]

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    Y = tf.convert_to_tensor(Y, dtype=tf.float32)
    return (X, Y)
        
def main():
    
    X, Y = load_dataset("../input/nexraddata/", 'data.npy')
    model = EncoderDecoder(
        2,
        [64, 48], [(3, 3), (3, 3)],
        16,
        (X.shape[2], X.shape[3], X.shape[4]),
        './training_checkpoints'
    )
    # model.restore()
    model.train(X[:700], Y[:700], 400, X[700:800], Y[700:800])

    test_model(model, X, Y)

if __name__ == "__main__":
    main()
