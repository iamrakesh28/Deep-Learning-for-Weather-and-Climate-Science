import tensorflow as tf

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
        self.batch_norm = []
        
        # volume convolution for the time step outputs
        # 1 x 1 CNN (patch size -> 4 x 4)
        self.conv_nn    = tf.keras.layers.Conv2D(filters=out_channel, 
                                                kernel_size=(1, 1),
                                                padding="same",
                                                activation="sigmoid",
                                                data_format="channels_last")
        
        # ConvLSTM layers and Batch Normalization
        for layer in range(self.dec_layers):
            lstm = tf.keras.layers.ConvLSTM2D(filters=self.unit_list[layer],
                                              kernel_size=self.filter_sz[layer],
                                              padding="same",
                                              return_state=True,
                                              data_format="channels_last")
            
            # norm = tf.keras.layers.BatchNormalization()
            # self.batch_norm.append(norm)
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
            # input_  = self.batch_norm[layer](output, training=training)
            # input_  = tf.expand_dims(input_, 1)
            input_  = tf.expand_dims(output, 1)
        
        frames = self.conv_nn(output)
        return frames, new_states
