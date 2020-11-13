# Next-frame prediction with Conv-LSTM
## Stacked Conv-LSTM
This is a sample tutorial code from https://keras.io/examples/vision/conv_lstm/ for the demonstration of next frame prediction with Conv-LSTM. The model has 3 stacked Conv-LSTM layers with 40 hidden units (filters of size (3 x 3)) in each layer. The final layer is a 1 filter 3D Convolutional layer with filter size (3 x 3 x 3) with sigmoid activation to produce the final image with intensities between 0 and 1. Between each two layers there is a batch normalization layer to make network stable and faster. The model is many-to-one which means given any number sequences, the model will predict the next frame of the given sequence. Artificial dataset is generated where each frame has 3 - 7 squares moving linearly over time inside the 40 x 40 image frame. Binary cross-entropy loss and Adadelta optimizer were used for the training. 

To run this model:

` python3 next-frame.py
`
### Training
The model was trained on 950 samples and 50 validation samples for 5 epochs.
```
Train on 950 samples, validate on 50 samples
Epoch 1/5
950/950 [==============================] - 37s 39ms/step - loss: 0.2410 - val_loss: 0.0775
Epoch 2/5
950/950 [==============================] - 29s 30ms/step - loss: 0.0235 - val_loss: 0.0157
Epoch 3/5
950/950 [==============================] - 29s 30ms/step - loss: 0.0044 - val_loss: 0.0042
Epoch 4/5
950/950 [==============================] - 29s 30ms/step - loss: 0.0015 - val_loss: 0.0019
Epoch 5/5
950/950 [==============================] - 29s 30ms/step - loss: 8.3291e-04 - val_loss: 7.3238e-04
CPU times: user 2min 21s, sys: 38.4 s, total: 3min
Wall time: 2min 36s
```

### Result
On a test sample, next 8 frames were predicted using first 7 frames. Since the above model is many-to-one, so for the many-to-many predictions, predicted output frames 
are fed into the network along with the original input sequence.
 ![Predicted](https://github.com/iamrakesh28/Deep-Learning-for-Weather-and-Climate-Science/blob/master/lstm-keras/next-frame-tutorial/images/output.gif) 
 
## Encoder-Decoder Model

One layer encoder-decoder model was used for training. The layer (both encoder and decoder) has 128 hidden units (no. of filters) with filter size of (3 x 3). The final layer in decoder is 1 x 1 2D conolutional layer with sigmoid activation to produce the frame with intensities between 0 and 1. Since it's a encoder-decoder model, the hidden states from encoder layer is used to initialize the hidden states in the decoder layer. The loss is propagated from the decoder prediction loss (encoder outputs are discarded). The training set is same as in the above model. Binary cross-entropy loss and RMSprop optimizer (learning rate = 0.001 and rho = 0.9) were used for the training. 

To run this model:

` python3 encoder-decoder.py
`
### Training
The model was trained on 1000 samples for 20 epochs with batch size of 8.
```
Epoch 1 Loss 0.1155
Time taken for 1 epoch 40.9735209941864 sec

Epoch 3 Loss 0.0074
Time taken for 1 epoch 36.06649732589722 sec

Epoch 5 Loss 0.0021
Time taken for 1 epoch 35.54285788536072 sec

Epoch 7 Loss 0.0010
Time taken for 1 epoch 36.459315061569214 sec

Epoch 9 Loss 0.0006
Time taken for 1 epoch 37.46419548988342 sec

Epoch 11 Loss 0.0004
Time taken for 1 epoch 36.94618797302246 sec

Epoch 13 Loss 0.0003
Time taken for 1 epoch 35.945321559906006 sec

Epoch 15 Loss 0.0002
Time taken for 1 epoch 36.00389909744263 sec

Epoch 17 Loss 0.0002
Time taken for 1 epoch 36.477062463760376 sec

Epoch 19 Loss 0.0001
Time taken for 1 epoch 37.519431352615356 sec

```

### Result

## References
[1] https://keras.io/examples/vision/conv_lstm/

[2] https://www.tensorflow.org/tutorials/text/nmt_with_attention
