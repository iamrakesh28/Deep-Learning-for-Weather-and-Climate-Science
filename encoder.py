import tensorflow as tf

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
            # if layer != self.enc_layers - 1:
            #    input_ = self.batch_norm[layer](outputs, training=training)
                
            states.append([hidden_state, cell_state])
        
        return states
