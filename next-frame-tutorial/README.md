# Next-frame prediction with Conv-LSTM
A simple Artificial dataset is generated where each frame has 3 - 7 squares moving linearly over time inside the 40 x 40 image frame. Each sample has 20 sequence of such image frames. The dataset generation code can be found at https://keras.io/examples/vision/conv_lstm/.

## Stacked Conv-LSTM
The model has 3 stacked Conv-LSTM layers with 40 hidden units (filters of size (3 x 3)) in each layer. The final layer is a 1 filter 3D Convolutional layer with filter size (1 x 1 x 1) having sigmoid activation to produce the final image with intensities between 0 and 1. Between each two layers there is a batch normalization layer to make network stable and faster. The model is many-to-one which means given any number sequences, the model will predict the next frame of the given sequence. The model was trained to minimize the binary crossentropy loss and Adadelta optimizer was used during the training. 

To run this model, set the epoch to 50. If the epoch is set higher then ensure that model doesn't overfit.

` python3 next-frame.py
`
### Training
The model was trained on 950 samples and 50 validation samples for 5 epochs.
```
Train on 950 samples, validate on 50 samples
Epoch 50/50
900/900 [==============================] - 30s 33ms/step - loss: 0.0055 - val_loss: 0.0053
CPU times: user 18min 27s, sys: 8min 21s, total: 26min 48s
```

### Result
On a test sample, next 10 frames were predicted using first 10 frames. Since the above model is many-to-one, so for the many-to-many predictions, predicted output frames are also feed into the network along with the original input sequence.

 ![input_stack](https://github.com/iamrakesh28/Deep-Learning-for-Weather-and-Climate-Science/blob/master/next-frame-tutorial/images/stack/input.gif)
 ![Predicted_stack](https://github.com/iamrakesh28/Deep-Learning-for-Weather-and-Climate-Science/blob/master/next-frame-tutorial/images/stack/output.gif) 
 <br /> 
 **An example from the test set:** Left image is the input frames and right image is the predicted frames vs the ground truth frames
## Encoder-Decoder Model

One layer encoder-decoder model was used for training. The layer (both encoder and decoder) has 128 hidden units (no. of filters) with filter size of (3 x 3). The final layer in decoder is 1 x 1 2D conolutional layer with sigmoid activation to produce the frame with intensities between 0 and 1. Since it's a encoder-decoder model, the hidden states from encoder layer is used to initialize the hidden states in the decoder layer. The loss is propagated from the decoder prediction loss (encoder outputs are discarded). The training set is same as in the above model. The model was trained to minimize the binary crossentropy loss and RMSprop optimizer (learning rate = 0.001 and rho = 0.9) was used during the training. 

To run this model, set the epoch to 20 or little higher. The model converges fastly.

` python3 encoder-decoder.py
`
### Training
The model was trained on 1000 samples for 20 epochs with batch size of 8.
```
Epoch 19 Loss 0.0001
Time taken for 1 epoch 37.519431352615356 sec

```

### Result
The model was used to predict 10 output frames sequence using 10 input seqeunce frames.  <br />
![input0](https://github.com/iamrakesh28/Deep-Learning-for-Weather-and-Climate-Science/blob/master/next-frame-tutorial/images/enc_dec0/input.gif)
![Predicted0](https://github.com/iamrakesh28/Deep-Learning-for-Weather-and-Climate-Science/blob/master/next-frame-tutorial/images/enc_dec0/output.gif) <br />
![input1](https://github.com/iamrakesh28/Deep-Learning-for-Weather-and-Climate-Science/blob/master/next-frame-tutorial/images/enc_dec1/input.gif)
![Predicted1](https://github.com/iamrakesh28/Deep-Learning-for-Weather-and-Climate-Science/blob/master/next-frame-tutorial/images/enc_dec1/output.gif)
<br />
**Examples from the test set:** Left image is the input frames and right image is the actual frames vs the predicted frames. <br />

## References
[1] https://keras.io/examples/vision/conv_lstm/ <br />
[2] https://www.tensorflow.org/tutorials/text/nmt_with_attention
