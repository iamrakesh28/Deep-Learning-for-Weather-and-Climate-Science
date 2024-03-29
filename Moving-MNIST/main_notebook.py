import tensorflow as tf
import numpy as np
import time
import os

class Encoder(tf.keras.Model):
    
    # unit_list -> list of units in each layer
    # filter_sz -> list of filter sizes for each layer
    def __init__(self, enc_layers, unit_list, filter_sz, image_sz, batch_sz):
        super(Encoder, self).__init__()
        
        self.enc_layers = enc_layers
        self.unit_list  = unit_list
        self.filter_sz  = filter_sz
        self.image_sz   = image_sz
        self.batch_sz   = batch_sz
        self.conv_lstm  = []
        # self.batch_norm = []
        
        for layer in range(self.enc_layers):
            lstm = tf.keras.layers.ConvLSTM2D(filters=self.unit_list[layer],
                                              kernel_size=self.filter_sz[layer],
                                              padding="same",
                                              return_sequences=True,
                                              return_state=True,
                                              data_format="channels_last")
            
            #if layer != self.enc_layers - 1:
            #    norm = tf.keras.layers.BatchNormalization()
            #    self.batch_norm.append(norm)
            self.conv_lstm.append(lstm)
            
    
    def initialize_states(self, layer, batch_sz):
        return [tf.zeros([batch_sz, self.image_sz[0], self.image_sz[1], self.unit_list[layer]]), 
                tf.zeros([batch_sz, self.image_sz[0], self.image_sz[1], self.unit_list[layer]])]
    
    
    # Encoder doesn't need states input
    # x.shape -> (batch_size, time_steps, rows, cols, channels)
    def call(self, input_, batch_sz, training=True):
        
        states = []
        for layer in range(self.enc_layers):
            outputs, hidden_state, cell_state = self.conv_lstm[layer](
                input_, 
                initial_state = self.initialize_states(layer, batch_sz)
            )
            input_ = outputs
            
            # No batch normalization for now
            # if layer != self.enc_layers - 1:
            #    input_ = self.batch_norm[layer](outputs, training=training)
                
            states.append([hidden_state, cell_state])
        
        return states

    
class Decoder(tf.keras.Model):
    
    # unit_list -> list of units in each layer
    # filter_sz -> list of filter sizes for each layer
    # keep parameters same as Encoder
    def __init__(self, dec_layers, unit_list, filter_sz, out_channel):
        super(Decoder, self).__init__()
        
        self.dec_layers = dec_layers
        self.unit_list  = unit_list
        self.filter_sz  = filter_sz
        self.conv_lstm  = []
        # self.batch_norm = []
        
        # volume convolution for the time step outputs
        # 1 x 1 CNN (patch size -> 4 x 4)
        self.conv_nn    = tf.keras.layers.Conv2D(filters=out_channel, 
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
            
            #norm = tf.keras.layers.BatchNormalization()
            #self.batch_norm.append(norm)
            self.conv_lstm.append(lstm)
    
    # input_.shape -> (batch_size, time_steps, rows, cols, channels)
    def call(self, input_, states, training=True):
        
        new_states = []
        for layer in range(self.dec_layers):
            output, hidden_state, cell_state = self.conv_lstm[layer](
                input_,
                initial_state=states[layer]
            )
            new_states.append([hidden_state, cell_state])
            #input_  = self.batch_norm[layer](output, training=training)
            #input_  = tf.expand_dims(input_, 1)
            input_  = tf.expand_dims(output, 1)
        
        frames = self.conv_nn(output)
        return frames, new_states
    
# Builds an encoder-decoder
class EncoderDecoder:
    def __init__(
        self, 
        num_layers, 
        unit_list, 
        filter_sz, 
        batch_sz, 
        image_sz,
        checkpoint_dir,
    ):
        self.num_layers  = num_layers
        self.batch_sz    = batch_sz
        self.image_sz    = image_sz
        self.encoder     = Encoder(num_layers, unit_list, filter_sz, image_sz, batch_sz)
        self.decoder     = Decoder(num_layers, unit_list, filter_sz, image_sz[2])
        self.optimizer   = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, 
            encoder=self.encoder, 
            decoder=self.decoder
        )
        
        # Binary crossentropy
        # T * logP + (1 - T) * log(1 - P)
        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        # self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        # self.loss_object = tf.keras.losses.CategoricalCrossentropy(
        #    reduction=tf.keras.losses.Reduction.SUM
        #)
        
    def __loss_function(self, real_frame, pred_frame):
        return self.loss_object(real_frame, pred_frame)
        
    # input_ -> (batch_size, time_steps, rows, cols, channels)
    # target -> (batch_size, time_steps, rows, cols, channels)
    def __train_step(self, input_, target):
        batch_loss = 0
        start_pred = input_.shape[1] - 1

        with tf.GradientTape() as tape:

            dec_states = self.encoder(input_[:, :start_pred, :, :, :], self.batch_sz, True)
            dec_input = tf.expand_dims(input_[:, start_pred, :, :, :], 1)
            
            # Teacher forcing
            for t in range(0, target.shape[1]):
                prediction, dec_states = self.decoder(dec_input, dec_states)
                
                batch_loss += self.__loss_function(target[:, t, :, :, :], prediction)
                
                # using teacher forcing
                dec_input = tf.expand_dims(target[:, t, :, :, :], 1)
        

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(batch_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return (batch_loss / int(target.shape[1]))
    
    # inputX - > (total, time_steps, rows, cols, channels)
    # targetY -> (total, time_steps, rows, cols, channels)
    def train(self, inputX, targetY, epochs, valX, valY, X, Y):
        init_time = time.time()
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
                
                batch_loss = self.__train_step(input_, target)
                total_loss += batch_loss
                
            # saving (checkpoint) the model every 25 epochs
            if epoch % 25 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                val_loss = self.evaluate(valX, valY)
                print('Epoch {} Evaluation Loss {:.4f}'.format(epoch + 1, val_loss))
                # if epoch % 50 == 0:
                test_model(self, X, Y)
                if (time.time() - init_time) / 3600.0 > 8:
                    break

            total_batch += 1
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / total_batch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            
    def restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
    
    # input_ -> (batch_size, time_steps, rows, cols, channels)
    # target -> (batch_size, time_steps, rows, cols, channels)
    # valid  -> validation
    def __eval_step(self, input_, target, valid):
        
        batch_loss = 0
        start_pred = input_.shape[1] - 1

        dec_states = self.encoder(input_[:, :start_pred, :, :, :], self.batch_sz, True)
        dec_input = tf.expand_dims(input_[:, start_pred, :, :, :], 1)
            
        for t in range(0, target.shape[1]):
            prediction, dec_states = self.decoder(dec_input, dec_states)    
            batch_loss += self.__loss_function(target[:, t, :, :, :], prediction)

            # if evaluating on validation set
            if valid:
                # using teacher forcing
                dec_input = tf.expand_dims(target[:, t, :, :, :], 1)
            else:
                # evaluating on testing set
                dec_input = tf.expand_dims(prediction, 1)
        
        return (batch_loss / int(target.shape[1]))

    # input -> (time_steps, rows, cols, channels)
    def predict(self, input_, output_seq):
        input_ = tf.expand_dims(input_, 0)
        start_pred = input_.shape[1] - 1
        dec_states = self.encoder(input_[:, :start_pred, :, :, :], 1, False)
        dec_input = tf.expand_dims(input_[:,-1, :, :, :], 1)
        
        predictions = []
        
        for t in range(output_seq):
            prediction, dec_states = self.decoder(dec_input, dec_states, False)
            dec_input = tf.expand_dims(prediction, 0)
            predictions.append(prediction.numpy().reshape(self.image_sz))
            
        return np.array(predictions)
    
    def evaluate(self, inputX, outputY, valid=True):
        
        total_loss = 0
        total_batch = inputX.shape[0] // self.batch_sz
        
        for batch in range(total_batch):
            index = batch * self.batch_sz
            input_ = inputX[index:index + self.batch_sz, :, :, :, :]
            target = outputY[index:index + self.batch_sz, :, :, :, :]
                
            batch_loss = self.__eval_step(input_, target, valid)
            total_loss += batch_loss
    
        total_batch += 1
        return total_loss / total_batch

def reshape_patch(data, patch_sz):
    data_patch = []
    for sample in range(data.shape[0]):
        
        data_patch.append([])
        
        for frame in range(data.shape[1]):
            
            data_patch[sample].append([])
            rows = data.shape[2] // patch_sz[0]
            
            for row in range(rows):
                
                data_patch[sample][frame].append([])
                cols = data.shape[3] // patch_sz[1]
                
                for col in range(cols):
                    
                    patch = data[sample][frame][
                        row * patch_sz[0] : (row + 1) * patch_sz[0],
                        col * patch_sz[1] : (col + 1) * patch_sz[1]
                    ]
                    
                    # better to use list() compared to patch.tolist() here
                    data_patch[sample][frame][row].append(list(patch.reshape(patch_sz[0] * patch_sz[1])))
        
    return np.array(data_patch)

def restore_patch(data, patch_sz):
    data_restore = np.zeros((data.shape[0], data.shape[1] * patch_sz[0], data.shape[2] * patch_sz[1]))
    
    for frame in range(data.shape[0]):
        for row in range(data.shape[1]):
            for col in range(data.shape[2]):
                patch = data[frame][row][col].reshape(patch_sz)
                data_restore[frame][
                    row * patch_sz[0] : (row + 1) * patch_sz[0],
                    col * patch_sz[1] : (col + 1) * patch_sz[1]
                ] = patch
    
    return data_restore

def load_dataset(path, filename):
    train_data = np.load(path + filename)
    # train_data = train_data.swapaxes(0, 1)
    train_data[[1005, 9000]] = train_data[[9000, 1005]]

    # patch size 4 x 4
    # train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 16, 16, 16)
    # train_data = reshape_patch(train_data, (4, 4))
    # plt.imshow(restore_patch(train_data[0], (4, 4))[0])
    # plt.show()
    
    train_data[train_data < 128] = 0
    train_data[train_data >= 128] = 1
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
    test_loss = model.evaluate(X[8500:], Y[8500:], False)
    print('Test Loss {:.4f}'.format(test_loss))

    y1 = model.predict(X[50], 10)
    y2 = model.predict(X[9000], 10)
    y3 = model.predict(X[9500], 10)
    y4 = model.predict(X[9345], 10)

    plot_result(
        restore_patch(X[50].numpy(), (4, 4)),
        restore_patch(Y[50].numpy(), (4, 4)),
        restore_patch(y1, (4, 4))
    )
    
    plot_result(
        restore_patch(X[9000].numpy(), (4, 4)),
        restore_patch(Y[9000].numpy(), (4, 4)),
        restore_patch(y2, (4, 4))
    )
    
    plot_result(
        restore_patch(X[9500].numpy(), (4, 4)),
        restore_patch(Y[9500].numpy(), (4, 4)),
        restore_patch(y3, (4, 4))
    )
    
    plot_result(
        restore_patch(X[9345].numpy(), (4, 4)),
        restore_patch(Y[9345].numpy(), (4, 4)),
        restore_patch(y4, (4, 4))
    )
        
def main():
    
    X, Y = load_dataset("../input/mnistreshape/", 'mnist-reshape.npy')
    
    model = EncoderDecoder(
        2,
        [128, 128], [(5, 5), (5, 5)],
        32,
        (X.shape[2], X.shape[3], X.shape[4]),
        './training_checkpoints'
    )
    #model.restore()
    model.train(X[:7000], Y[:7000], 200, X[7000:8500], Y[7000:8500], X, Y)

    test_model(model, X, Y)
    

if __name__ == "__main__":
    main()

