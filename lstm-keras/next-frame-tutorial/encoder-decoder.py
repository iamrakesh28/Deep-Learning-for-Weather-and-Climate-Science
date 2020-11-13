import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# Generate movies with 3 to 7 moving squares inside.
# visit : https://keras.io/examples/vision/conv_lstm/ for details
# @param n_samples number of samples to generate
# @param n_frames number of sequences in each sample
def generate_movies(n_samples=1200, n_frames=20):
    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1),
                              dtype=np.float)

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

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[i, t, x_shift - w: x_shift + w,
                               y_shift - w: y_shift + w, 0] += 1

    # Cut to a 40x40 window
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies[shifted_movies >= 1] = 1
    return shifted_movies


# Encoder model to encapsulate the input sequences
class Encoder(tf.keras.Model):

    # @param enc_layers number of conv lstm layers
    # @param unit_list list of hidden units in each conv lstm layer
    # @param filter_sz list of filter sizes for each layer
    def __init__(self, enc_layers, unit_list, filter_sz):
        super(Encoder, self).__init__()
        
        self.enc_layers = enc_layers
        self.unit_list  = unit_list
        self.filter_sz  = filter_sz
        self.conv_lstm  = []
        self.batch_norm = []
        
        for layer in range(self.enc_layers):
            # conv lstm layer
            lstm = tf.keras.layers.ConvLSTM2D(filters=self.unit_list[layer],
                                              kernel_size=self.filter_sz[layer],
                                              padding="same",
                                              return_sequences=True,
                                              return_state=True,
                                              data_format="channels_last")

            # batch normalization layer after each conv lstm except after the last layer
            if layer != self.enc_layers - 1:
                norm = tf.keras.layers.BatchNormalization()
                self.batch_norm.append(norm)
            self.conv_lstm.append(lstm)
            
    # Encoder doesn't need states input
    # input_.shape -> (batch_size, time_steps, rows, cols, channels)
    # @return list of pairs of states from each layer
    def call(self, input_, training=True):
        
        states = []
        for layer in range(self.enc_layers):
            outputs, hidden_state, cell_state = self.conv_lstm[layer](input_)
            
            if layer != self.enc_layers - 1:
                input_ = self.batch_norm[layer](outputs, training=training)
                
            states.append([hidden_state, cell_state])
        
        return states
    

# Decode the state inputs from encoder and predicts the sequence output
class Decoder(tf.keras.Model):

    # @param dec_layers number of conv lstm layers
    # @param unit_list -> list of hidden units in each conv lstm layer
    # @param filter_sz -> list of filter sizes for each conv lstm layer
    # Note : keep parameters same as Encoder
    def __init__(self, dec_layers, unit_list, filter_sz):
        super(Decoder, self).__init__()
        
        self.dec_layers = dec_layers
        self.unit_list  = unit_list
        self.filter_sz  = filter_sz
        self.conv_lstm  = []
        self.batch_norm = []
        
        # 2D convolution for the time step outputs
        # 1 x 1 CNN
        self.conv_nn    = tf.keras.layers.Conv2D(filters=1, 
                                                kernel_size=(1, 1),
                                                padding="same",
                                                activation='sigmoid',
                                                data_format="channels_last")
        
        # ConvLSTM layers and Batch Normalization
        for layer in range(self.dec_layers):
            lstm = tf.keras.layers.ConvLSTM2D(filters=self.unit_list[layer],
                                              kernel_size=self.filter_sz[layer],
                                              padding="same",
                                              return_state=True,
                                              data_format="channels_last")

            # batch normalization layer after each conv lstm layer
            norm = tf.keras.layers.BatchNormalization()
            self.batch_norm.append(norm)
            self.conv_lstm.append(lstm)
    
    # input_.shape -> (batch_size, time_steps, rows, cols, channels)
    # @param states hidden and cell states to be fed to the Decoder
    # @return predicted frame and list of states from each layer
    def call(self, input_, states, training=True):
        
        new_states = []
        for layer in range(self.dec_layers):
            output, hidden_state, cell_state = self.conv_lstm[layer](
                input_,
                initial_state=states[layer]
            )
            new_states.append([hidden_state, cell_state])
            input_  = self.batch_norm[layer](output, training=training)
            input_  = tf.expand_dims(input_, 1)
        
        frames = self.conv_nn(output)
        return frames, new_states
    
    '''
    def initialize_states(self):
        return [tf.zeros([self.batch_sz, self.image_sz[0], self.image_sz[1], self.units]), 
                tf.zeros([self.batch_sz, self.image_sz[0], self.image_sz[1], self.units])]
    '''
    
# Builds an Encoder-Decoder model
class EncoderDecoder:
    
    # @param num_layers number of layers in the model
    # @param unit_list list of hidden states size for each layer
    # @param filter_sz list of filter sizes for each layer
    # @param batch_sz batch size
    # @param image_sz image size (gray scale)
    # Note : Keep encoder and decoder layer parameters same
    def __init__(self, num_layers, unit_list, filter_sz, batch_sz, image_sz):
        self.num_layers  = num_layers
        # encoder layer defined
        self.encoder     = Encoder(num_layers, unit_list, filter_sz)
        # decoder layer defined
        self.decoder     = Decoder(num_layers, unit_list, filter_sz)
        self.batch_sz    = batch_sz
        # RMS optimizer
        self.optimizer   = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
        self.image_sz    = image_sz
        
        # Binary crossentropy loss (average)
        # Sigma T * logP + (1 - T) * log(1 - P)
        self.loss_object = tf.keras.losses.BinaryCrossentropy()

    # @return average loss
    def loss_function(self, real_frame, pred_frame):
        return self.loss_object(real_frame, pred_frame)

    # Trains for one iteration
    # @param input_ input sequences
    # @param target target sequences
    # input_.shape -> (batch_size, time_steps, rows, cols, channels)
    # target .shape-> (batch_size, time_steps, rows, cols, channels)
    
    def train_step(self, input_, target):
        batch_loss = 0

        with tf.GradientTape() as tape:
            dec_states = self.encoder(input_[:, :9, :, :, :])
            dec_input = tf.expand_dims(input_[:, 9, :, :, :], 1)
            
            # Teacher forcing
            for t in range(0, target.shape[1]):
                prediction, dec_states = self.decoder(dec_input, dec_states)
                
                batch_loss += self.loss_function(target[:, t, :, :, :], prediction)
                
                # using teacher forcing
                dec_input = tf.expand_dims(target[:, t, :, :, :], 1)
        

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        # back propagates the gradient
        gradients = tape.gradient(batch_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        # batch_loss (average loss over each target output)
        return loss / int(targ.shape[1])
    
    # inputX - > (total, time_steps, rows, cols, channels)
    # targetY -> (total, time_steps, rows, cols, channels)
    def train(self, inputX, targetY, epochs):
        
        assert(inputX.shape == targetY.shape)
        
        for epoch in range(epochs):
            start = time.time()
            total_loss = 0
            total_batch = inputX.shape[0] // self.batch_sz
            #print(total_batch)
            
            for batch in range(total_batch):
                index = batch * self.batch_sz
                input_ = inputX[index:index + self.batch_sz, :, :, :, :]
                target = targetY[index:index + self.batch_sz, :, :, :, :]
                
                # print(input_.shape, target.shape)
                
                batch_loss = self.train_step(input_, target)
                total_loss += batch_loss
                
                #if batch % 10 == 0:
                #    print('Epoch {} Batch {} Loss {:.4f}'
                #          .format(epoch + 1, batch + 1, batch_loss.numpy()))
  
            # saving (checkpoint) the model every 2 epochs
            if (epoch) % 2 == 0:
            #    checkpoint.save(file_prefix = checkpoint_prefix)
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / self.batch_sz))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
                
    # input -> (time_steps, rows, cols, channels)
    def predict(self, input_, output_seq):
        input_ = tf.expand_dims(input_, 0)
        dec_states = self.encoder(input_[:9, :, :, :], False)
        dec_input = tf.expand_dims(input_[-1, :, :, :], 0)
        
        predictions = []
            

        # Teacher forcing
        for t in range(output_seq):
            prediction, dec_states = self.decoder(dec_input, dec_states, False)
            dec_input = tf.expand_dims(prediction, 0)
            predictions.append(prediction.numpy().reshape(self.image_sz))
            
        return np.array(predictions)
        
    
def load_dataset():
    path = "../input/mnist-cs/"
    data = np.load(path + 'mnist_test_seq.npy')
    data = data.swapaxes(0, 1)
    train_data = data[:105]
    train_data[train_data < 128] = 0
    train_data[train_data >= 128] = 1
    train_data = np.expand_dims(train_data, 4)
    #print(train_data.shape)
    X = train_data[:, :10, :, :, :]
    Y = train_data[:, 10:21, :, :, :]
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    Y = tf.convert_to_tensor(Y, dtype=tf.float32)
    return (X, Y)
    

shifted_movies = generate_movies(n_samples=1200)
#for i in range(15):
#    plt.imshow(shifted_movies[0][i].reshape(40, 40))
#    plt.show()

def plot_result(input_, predict):
    for i in range(input_.shape[0]):
        plt.imshow(input_[i])
        plt.title("Actual_" + str(i + 1))
        plt.show()
    for i in range(predict.shape[0]):
        plt.imshow(predict[i])
        plt.title("Predicted_" + str(i + 11))
        plt.show()
        
def main():
    
    
    #X, Y = load_dataset()
    print(shifted_movies.shape)
    
    #print(X.shape, Y.shape)
    X = shifted_movies[:, :10, :, :, :]
    Y = shifted_movies[:, 10:, :, :, :]
    model = EncoderDecoder(1, [128], [(3, 3)], 8, (X.shape[2], X.shape[3]))
    model.train(X[:1000], Y[:1000], 20)
    y1 = model.predict(X[1100], 10)
    y2 = model.predict(X[1005], 10)
    plot_result(X[1100].reshape(10, 40, 40), y1)
    plot_result(X[1005].reshape(10, 40, 40), y2)
    

    
if __name__ == "__main__":
    main()
