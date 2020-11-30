import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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
        self.batch_norm = []
        
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
            #if layer != self.enc_layers - 1:
            #    input_ = self.batch_norm[layer](outputs, training=training)
                
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
        self.batch_norm = []
        
        # volume convolution for the time step outputs
        # 1 x 1 CNN (patch size -> 4 x 4)
        self.conv_nn    = tf.keras.layers.Conv2D(filters=4, 
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
        self.decoder     = Decoder(num_layers, unit_list, filter_sz)
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
        
    def loss_function(self, real_frame, pred_frame):
        return self.loss_object(real_frame, pred_frame)
        
    # input_ -> (batch_size, time_steps, rows, cols, channels)
    # target -> (batch_size, time_steps, rows, cols, channels)
    def train_step(self, input_, target):
        batch_loss = 0
        start_pred = input_.shape[1] - 1

        with tf.GradientTape() as tape:

            dec_states = self.encoder(input_[:, :start_pred, :, :, :], self.batch_sz, True)
            dec_input = tf.expand_dims(input_[:, start_pred, :, :, :], 1)
            
            # Teacher forcing
            for t in range(0, target.shape[1]):
                prediction, dec_states = self.decoder(dec_input, dec_states)
                
                batch_loss += self.loss_function(target[:, t, :, :, :], prediction)
                
                # using teacher forcing
                dec_input = tf.expand_dims(target[:, t, :, :, :], 1)
        

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(batch_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return (batch_loss / int(target.shape[1]))
    
    # inputX - > (total, time_steps, rows, cols, channels)
    # targetY -> (total, time_steps, rows, cols, channels)
    def train(self, inputX, targetY, epochs, X, Y):
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
                
                batch_loss = self.train_step(input_, target)
                total_loss += batch_loss
                
            # saving (checkpoint) the model every 5 epochs
            if epoch % 25 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                if epoch % 50 == 0:
                    self.test_model(X, Y)
                if (time.time() - init_time) / 3600.0 > 8:
                    break
            #    self.checkpoint.save(file_prefix = self.checkpoint_prefix)
            total_batch += 1
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / total_batch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            
    def restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        self.checkpoint_dir = "./training_checkpoints"
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, 
            encoder=self.encoder, 
            decoder=self.decoder
        )
                
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
            predictions.append(prediction.numpy().reshape(100, 100))
            
        return np.array(predictions)
    
    # input_ -> (batch_size, time_steps, rows, cols, channels)
    # target -> (batch_size, time_steps, rows, cols, channels)
    def eval_step(self, input_, target):
        
        batch_loss = 0
        start_pred = input_.shape[1] - 1

        dec_states = self.encoder(input_[:, :start_pred, :, :, :], self.batch_sz, True)
        dec_input = tf.expand_dims(input_[:, start_pred, :, :, :], 1)
            
        for t in range(0, target.shape[1]):
            prediction, dec_states = self.decoder(dec_input, dec_states)    
            batch_loss += self.loss_function(target[:, t, :, :, :], prediction)
            # using teacher forcing
            dec_input = tf.expand_dims(target[:, t, :, :, :], 1)
        
        return (batch_loss / int(target.shape[1]))
    
    def pred_step(self, input_, target):
        
        batch_loss = 0
        start_pred = input_.shape[1] - 1

        dec_states = self.encoder(input_[:, :start_pred, :, :, :], self.batch_sz, True)
        dec_input = tf.expand_dims(input_[:, start_pred, :, :, :], 1)
            
        for t in range(0, target.shape[1]):
            prediction, dec_states = self.decoder(dec_input, dec_states)    
            batch_loss += self.loss_function(target[:, t, :, :, :], prediction)
            dec_input = tf.expand_dims(prediction, 1)
        
        return (batch_loss / int(target.shape[1]))
    
    def evaluate(self, inputX, outputY, valid=True):
        
        total_loss = 0
        total_batch = inputX.shape[0] // self.batch_sz
        for batch in range(total_batch):
            index = batch * self.batch_sz
            input_ = inputX[index:index + self.batch_sz, :, :, :, :]
            target = outputY[index:index + self.batch_sz, :, :, :, :]
                
            if valid == True:
                batch_loss = self.eval_step(input_, target)
                total_loss += batch_loss
            else:
                batch_loss = self.pred_step(input_, target)
                total_loss += batch_loss
    
        total_batch += 1
        print('Evaluation: Total Loss {:.4f}'.format(total_loss / total_batch))
        return total_loss / total_batch
    
    def test_model(self, X, Y):
        e1 = self.evaluate(X[700:800], Y[700:800], True)
        e2 = self.evaluate(X[800:], Y[800:], False)
        y1 = self.predict(X[50], 10)
        y2 = self.predict(X[150], 10)
        y3 = self.predict(X[940], 10)
        y4 = self.predict(X[934], 10)
        plot_result(X[50].numpy().reshape(10, 100, 100), Y[50].numpy().reshape(10, 100, 100), y1)
        plot_result(X[150].numpy().reshape(10, 100, 100), Y[150].numpy().reshape(10, 100, 100), y2)
        plot_result(X[940].numpy().reshape(10, 100, 100), Y[940].numpy().reshape(10, 100, 100), y3)
        plot_result(X[934].numpy().reshape(10, 100, 100), Y[934].numpy().reshape(10, 100, 100), y4)

    
def load_dataset(path, filename):
    train_data = np.load(path + filename)
    # patch size 4 x 4
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 50, 50, 4)
    train_data[train_data < 200] = 0
    train_data[train_data >= 200] = 1
    #train_data = train_data / 255.0
    print(train_data.min(), train_data.max())
    # train_data = np.expand_dims(train_data, 4)
    print(train_data.shape)
    X = train_data[:, :10, :, :, :]
    Y = train_data[:, 10:21, :, :, :]
    plt.show()
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

X, Y = load_dataset("../input/nexraddata/", 'data.npy')
model = EncoderDecoder(2, [64, 48], [(3, 3), (3, 3)], 16, (X.shape[2], X.shape[3]), './training_checkpoints')
#model.restore()
model.train(X[:700], Y[:700], 400, X, Y)

model.test_model(X, Y)

