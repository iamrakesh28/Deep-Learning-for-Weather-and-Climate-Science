# NEXRAD
The Next Generation Weather Radar (NEXRAD) system currently comprises 160 sites throughout the United States and select overseas locations.
NEXRAD detects precipitation and atmospheric movement or wind. It returns data which when processed can be displayed in a mosaic map which 
shows patterns of precipitation and its movement. The NCEI archive includes the base data, called Level-II, and the derived products, 
called Level-III. Level-II data include the original three meteorological base data quantities: reflectivity, mean radial velocity, and 
spectrum width, as well as the dual-polarization base data of differential reflectivity, correlation coefficient, and differential phase.

Radar Data can be accessed at [https://www.ncdc.noaa.gov/nexradinv/]. There are different ways to access the data. For eg., Data can accessed 
by Single Site and Day, Multiple Sites and Days, etc.

## Download Data
Single day and single site Level-II can be downloaded directly (~500 MB per day). Other data such as Level-III products, multiple days, etc need to be ordered and processing takes around 1-2 hours. 
To directly download the data for single site and single day, change the GET request paramaeters for site, day, month and year in `download_day.py` and run
```
python3 download_day.py
```

## Reflexivity Plot
The reflexivity can be plotted using the libraries : MetPy and Py-Art.
MetPy:
```
python3 read_metpy.py
```
Py-Art:
```
python3 read_pyart.py
```
<p align="center">
  <img src="https://github.com/iamrakesh28/Deep-Learning-for-Weather-and-Climate-Science/blob/master/NEXRAD/radar_metpy.png" width=540>
  <img src="https://github.com/iamrakesh28/Deep-Learning-for-Weather-and-Climate-Science/blob/master/NEXRAD/KATX/katx.png" width=400>
  </br>
  <em> (a) Using MetPy </em>
  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
  <em> (b) Using Py-Art </em>
</p>

## Create Dataset
To extract the reflexivity plot from Level-II data, run
```
python3 dataset_day.py
```
It will download the data for a single day and save a numpy array for the plots with each frame size 100 x 100.
To create dataset for multiple days, run
```
python3 dataset_mult_day.py
```


## References
[1] (http://arm-doe.github.io/pyart/source/auto_examples/plotting/plot_nexrad_reflectivity.html) </br>
[2] (https://unidata.github.io/MetPy/latest/examples/formats/NEXRAD_Level_2_File.html)
