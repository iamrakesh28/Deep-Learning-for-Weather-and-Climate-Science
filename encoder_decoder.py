import tensorflow as tf
import numpy as np
import time

class Encoder(tf.keras.Model):
    
    # unit_list -> list of units in each layer
    # filter_sz -> list of filter sizes for each layer
    def __init__(self, enc_layers, unit_list, filter_sz):
        super(Encoder, self).__init__()
        
        self.enc_layers = enc_layers
        self.unit_list  = unit_list
        self.filter_sz  = filter_sz
        self.conv_lstm  = []
        
        for layer in range(self.enc_layers):
            lstm = tf.keras.layers.ConvLSTM2D(filters=self.unit_list[layer],
                                              kernel_size=self.filter_sz[layer],
                                              padding="same",
                                              return_sequences=True,
                                              return_state=True,
                                              data_format="channels_last")
            self.conv_lstm.append(lstm)
    
    # Encoder doesn't need states input
    # x.shape -> (batch_size, time_steps, rows, cols, channels)
    def call(self, input_):
        
        states = []
        for layer in range(self.enc_layers):
            outputs, hidden_state, cell_state = self.conv_lstm[layer](input_)
            input_ = outputs
            states.append([hidden_state, cell_state])
        
        return states
    
    
class Decoder(tf.keras.Model):
    
    # unit_list -> list of units in each layer
    # filter_sz -> list of filter sizes for each layer
    # keep parameters same as Encoder
    def __init__(self, dec_layers, unit_list, filter_sz):
        super(Decoder, self).__init__()
        
        self.dec_layers = dec_layers
        self.unit_list  = unit_list
        self.filter_sz  = filter_sz
        self.conv_lstm  = []
        
        # volume convolution for the time step outputs
        # 1 x 1 CNN
        self.conv_nn    = tf.keras.layers.Conv2D(filters=1, 
                                                kernel_size=(1, 1),
                                                padding="same",
                                                data_format="channels_last")
        
        # ConvLSTM layers
        for layer in range(self.dec_layers):
            lstm = tf.keras.layers.ConvLSTM2D(filters=self.unit_list[layer],
                                              kernel_size=self.filter_sz[layer],
                                              padding="same",
                                              return_state=True,
                                              data_format="channels_last")
            self.conv_lstm.append(lstm)
    
    # input_.shape -> (batch_size, time_steps, rows, cols, channels)
    def call(self, input_, states):
        
        new_states = []
        for layer in range(self.dec_layers):
            output, hidden_state, cell_state = self.conv_lstm[layer](input_,
                                                                     initial_state=states[layer])
            new_states.append([hidden_state, cell_state])
            input_  = output
        
        frames = self.conv_nn(output)
        return frames, new_states
    
    '''
    def initialize_states(self):
        return [tf.zeros([self.batch_sz, self.image_sz[0], self.image_sz[1], self.units]), 
                tf.zeros([self.batch_sz, self.image_sz[0], self.image_sz[1], self.units])]
    '''
    
# Builds an encoder-decoder
class EncoderDecoder:
    def __init__(self, num_layers, unit_list, filter_sz, batch_sz, image_sz):
        self.num_layers  = num_layers
        self.encoder     = Encoder(num_layers, unit_list, filter_sz)
        self.decoder     = Decoder(num_layers, unit_list, filter_sz)
        self.batch_sz    = batch_sz
        self.optimizer   = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
        self.image_sz    = image_sz
        
        # Binary crossentropy
        # T * logP + (1 - T) * log(1 - P)
        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        
    def loss_function(self, real_frame, pred_frame):
        return self.loss_object(real_frame, pred_frame)
        
    # input_ -> (batch_size, time_steps, rows, cols, channels)
    # target -> (batch_size, time_steps, rows, cols, channels)
    def train_step(self, input_, target):
        batch_loss = 0

        with tf.GradientTape() as tape:
            dec_states = self.encoder(input_)
            dec_input = tf.expand_dims(input_[:, -1, :, :, :], 1)
            
            # Teacher forcing
            for t in range(0, target.shape[1]):
                prediction, dec_states = self.decoder(dec_input, dec_states)
                
                batch_loss += self.loss_function(target[:, t, :, :, :], prediction)
                
                # using teacher forcing
                dec_input = tf.expand_dims(target[:, t, :, :, :], 1)
        
        # batch_loss = (loss / int(targ.shape[0]))

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(batch_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return batch_loss
    
    # inputX - > (total, time_steps, rows, cols, channels)
    # targetY -> (total, time_steps, rows, cols, channels)
    def train(self, inputX, targetY, epochs):
        
        assert(inputX.shape == targetY.shape)
        
        for epoch in range(epochs):
            start = time.time()
            total_loss = 0
            total_batch = inputX.shape[0] // self.batch_sz
            print(total_batch)
            
            for batch in range(total_batch):
                index = batch * self.batch_sz
                input_ = inputX[index:index + self.batch_sz, :, :, :, :]
                target = targetY[index:index + self.batch_sz, :, :, :, :]
                
                # print(input_.shape, target.shape)
                
                batch_loss = self.train_step(input_, target)
                total_loss += batch_loss

                if batch % 10 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'
                          .format(epoch + 1, batch + 1, batch_loss.numpy()))
  
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
            #    checkpoint.save(file_prefix = checkpoint_prefix)
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / self.batch_sz))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    
def load_dataset():
    path = ""
    data = np.load(path + 'mnist_test_seq.npy')
    data = data.swapaxes(0, 1)
    train_data = data[:100]
    train_data[train_data < 128] = 0
    train_data[train_data >= 128] = 1
    train_data = np.expand_dims(train_data, 4)
    #print(train_data.shape)
    X = train_data[:, :10, :, :, :]
    Y = train_data[:, 10:21, :, :, :]
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    Y = tf.convert_to_tensor(Y, dtype=tf.float32)
    return (X, Y)
    
def main():
    
    X, Y = load_dataset()
    print(X.shape, Y.shape)
    
    model = EncoderDecoder(1, [64], [(3, 3)], 8, (X.shape[2], X.shape[3]))
    model.train(X, Y, 1)
    '''
    encoder = Encoder(2, [4, 4], [(3, 3), (3, 3)])
    inputs = np.random.normal(size=(32, 10, 8, 8, 1))
    print(inputs.shape)
    states = encoder(inputs)
    print(type(states), type(states[0][0]))
    
    lstm = tf.keras.layers.ConvLSTM2D(filters=2,
                                      kernel_size=(3, 3),
                                      padding="same",
                                      return_sequences=True,
                                      return_state=True)
    out = lstm(inputs)
    print(out[0].shape)
    #[a, b] = encoder.initialize_states()
    #a, b, c = encoder(inputs, [a, b])
    #print(type(a), b.shape, c.shape)
    
    
    a = 0
    y_true = tf.constant([[[0], [1]]])
    y_pred = tf.constant([[[0.6], [0.6]]])
    # print(y_pred, y_true)
    # Using 'auto'/'sum_over_batch_size' reduction type.
    bce = tf.keras.losses.BinaryCrossentropy()
    print((bce(y_true, y_pred) + a))
    print(tf.reduce_mean([2, 2]))
    # print(encoder.trainable_variables)
    '''

    
if __name__ == "__main__":
    main()
