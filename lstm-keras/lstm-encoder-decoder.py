from numpy import array
from keras.models import Model
from keras.layers import Input, LSTM, Dense, RepeatVector

def createData(n):
    X = [x + 1 for x in range(n)]
    Y = [y * 15 for y in X]

    #print(X, Y)
    return (X, Y)

# encoder decoder model
# 50 - 50
if __name__ == "__main__":
    
    X, Y = createData(20)
    X = array(X).reshape(20, 1, 1)   # samples, time-steps, features
    Y = array(Y)

    # encoder LSTM
    encoder_inputs = Input(shape=(1, 1))   # time-steps, features
    encoder = LSTM(50, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # decoder LSTM
    # decoder_inputs = Inputs(shape=())
    # using repeat vector for now
    # repeats the encoder_outputs n (50) times
    '''
    decoder_inputs = RepeatVector(50)(encoder_outputs)
    #decoder_inputs = Input(shape=(1, 1))
    decoder_lstm = LSTM(50, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    '''
    decoder_dense = Dense(1, activation='softmax')
    decoder_outputs = decoder_dense(encoder_outputs)
    
    model = Model(encoder_inputs, decoder_outputs)
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    model.fit(X, Y, epochs=10, validation_split=0.2, batch_size=5)

    test = array([30]).reshape((1, 1, 1))
    output = model.predict(test, verbose=0)

    print(output)
