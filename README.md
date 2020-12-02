# Deep-Learning-for-Weather-and-Climate-Science

Nowcasting is weather forecasting on a very short term mesosacle period upto 6 hours. The goal is to give precise and timely prediction of precipitation, storm structure, hail potential, etc. in a local region over a short time period 
(eg., 0-6 hours). These predictions are useful for producing rainfall, storms, hails, etc alerts, providing weather guidance for airports, etc.

Weather Radar’s reflexivity is used by scientists to detect precipitation, evaluate storm structure, determine hail potential, etc. Sequence of radar reflexivity over a region for some time duration has spatiotemporal nature. Weather nowcasting is a spatiotemporal sequence forecasting problem with the sequence of past reflexivity maps as input and the
sequence of future reflexivity maps as output.

The LSTM encoder-decoder framework provides a general framework for sequence-to-sequence learning problems. I have implemented Convolutional LSTM Encoder-Decoder Network [1] for weather forecasting with the sequences being maps of 
radar reflexivity.

## Weather Forecasting using NEXRAD

The Next Generation Weather Radar (NEXRAD) [4] system currently comprises 160 sites throughout the United States and select overseas locations. NEXRAD detects precipitation and atmospheric movement or wind. It returns data which when processed can be displayed in a mosaic map which shows patterns of precipitation and its movement. NEXRAD Level-II
(Base) data include the original three meteorological base data quantities: reflectivity, mean radial velocity, and spectrum width. Data is collected from the radar sites usually at the interval of 4, 5, 6 or 10 minutes depending upon 
the volume coverage. Radar Data can be accessed at https://www.ncdc.noaa.gov/nexradinv/.

Reflexivity is expressed in dBZ. Higher value of reflexivity tells heavy precipiation or hail at that place and lower value tells light precipiation. For examples, 65 dBZ means extremely heavy precipitation (410 mm per hour, but likely hail), 50 dBZ means heavy precipitation (51 mm per hour), 35 dBZ tells moderate precipitation of 6.4 mm per hour [2], and 
so on. So, reflectivity component from the Level-II data can be used for weather forecasting for short duration.

In this project, weather forecasting was done for two regions : Seattle, WA and South Shore,
Hawaii. For each region, radar level-II data was collected for some duration and reflexivity plots
were extracted. These plots or images were resized into 100 x 100 images using nearest-neighbor
interpolation. Further, the images were converted to gray scale and later thresholded to have
binary intensities. These image sequences were later used for training and testing the models.
For each region, weather forecasting was done independently. (For simpler dataset, the images
were modified to have only two intensities and the model were trained to predict only the shapes
not the intensities)

### PHWA-SOUTH SHORE, HAWAII
PHWA is the id of radar at South Shore, Hawaii. A dataset of 959 sequences with 20 radar
maps or images in each sequence was created by collecting radar data of around 30 days from
August-2020 to October-2020. Time gap between each frame of a sequence was around 5
minutes. These sequences were separated into 700 training sequences, 100 validation sequences
and 159 test sequences. The following Encoder-Decoder networks were trained for forecasting.
 2 Layers with 64, 48 hidden units and (3 x 3) filter size in each layer. The input frames
were reshaped into 50 x 50 x 4 vectors. The average binary crossentropy loss was 0.1491.
 4 Layers with 96, 64, 64, 32 hidden units and (3 x 3) filter size in each layer. The input
frames were reshaped into 25 x 25 x 16 vectors. The average binary crossentropy loss was
0.1790.

Frames were reshaped to increase the channel size so that deeper models could be trained on
limited resources but increasing the frame channel size too much resulted in bad performance.
### KATX-SEATTLE, WA
Radar id at Seattle, WA is KATX. A dataset of 499 sequences with 20 radar maps in each
sequence was created by collecting radar data of around 30 days from January-2020 to April-
2020. Time gap between each frame of a sequence was around 10 minutes. These sequences
were separated into 350 training sequences, 75 validation sequences and 74 test sequences.
 4 Layers with 96, 96, 32, 32 hidden units and (3 x 3) filter size in each layer. The input
frames were reshaped into 25 x 25 x 16 vectors. The average binary crossentropy loss was
0.3761.
