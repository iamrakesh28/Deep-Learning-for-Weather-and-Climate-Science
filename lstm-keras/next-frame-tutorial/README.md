# Next-frame prediction with Conv-LSTM

## Training
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

## Result
On a test sample, next 8 frames were predicted using first 7 frames.
 ![Predicted](https://github.com/iamrakesh28/Deep-Learning-for-Weather-and-Climate-Science/tree/master/lstm-keras/next-frame-tutorial/images/output.gif) 
 
## References
[1] https://keras.io/examples/vision/conv_lstm/