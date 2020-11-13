# Next-frame prediction with Conv-LSTM
## Stacked Conv-LSTM
This is a sample tutorial code from https://keras.io/examples/vision/conv_lstm/ for the demonstration of next frame prediction with Conv-LSTM. The model has 3 stacked Conv-LSTM layers with 40 hidden units (filters of size (3 x 3)) in each layer. The final layer is a 1 filter 3D Convolutional layer to produce the final image. The model is many-to-one. Artificial dataset is generated where each frame has 3 - 7 squares moving linearly over time inside the 40 x 40 image frame.

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
On a test sample, next 8 frames were predicted using first 7 frames. Since the above model is many-to-one, so for the many-to-many predictions predicted output frame 
is also fed into the network along with the original input sequence.
 ![Predicted](https://github.com/iamrakesh28/Deep-Learning-for-Weather-and-Climate-Science/blob/master/lstm-keras/next-frame-tutorial/images/output.gif) 
 
## References
[1] https://keras.io/examples/vision/conv_lstm/
