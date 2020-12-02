import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from encoder_decoder import EncoderDecoder
    
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
    
def plot_result(input_, actual, predict):
    
    for i in range(input_.shape[0]):
        plt.imshow(input_[i])
        plt.title("Actual_" + str(i + 1))
        plt.show()
        
    for i in range(actual.shape[0]):
        plt.subplot(121), plt.imshow(actual[i]),
        plt.title("Actual_" + str(i + 1 + input_.shape[0]))
        plt.subplot(122), plt.imshow(predict[i]),
        plt.title("Predicted_" + str(i + 1 + input_.shape[0]))
        plt.show()
        
def test_model(model, X, Y):
    #e1 = model.evaluate(X[700:800], Y[700:800], True)
    test_loss = model.evaluate(X[800:], Y[800:], False)
    print('Test Loss {:.4f}'.format(test_loss))

    y1 = model.predict(X[50], 10)
    y2 = model.predict(X[150], 10)
    y3 = model.predict(X[940], 10)
    y4 = model.predict(X[934], 10)

    plot_result(X[50].numpy().reshape(10, 100, 100), Y[50].numpy().reshape(10, 100, 100), y1)
    plot_result(X[150].numpy().reshape(10, 100, 100), Y[150].numpy().reshape(10, 100, 100), y2)
    plot_result(X[940].numpy().reshape(10, 100, 100), Y[940].numpy().reshape(10, 100, 100), y3)
    plot_result(X[934].numpy().reshape(10, 100, 100), Y[934].numpy().reshape(10, 100, 100), y4)
        
def main():
    
    X, Y = load_dataset("../input/nexraddata/", 'data.npy')
    model = EncoderDecoder(
        2,
        [64, 48], [(3, 3), (3, 3)],
        16,
        (X.shape[2], X.shape[3], X.shape[4]),
        './training_checkpoints'
    )
    #model.restore()
    model.train(X[:700], Y[:700], 400, X[700:800], Y[700:800])

    model.test_model(X, Y)

if __name__ == "__main__":
    main()
