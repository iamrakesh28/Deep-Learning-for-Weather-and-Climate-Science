import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import pyart

def main():

    filename = "PHWA20201031_000332_V06"
    radar = pyart.io.read_nexrad_archive(filename)

    radar_np = radar.fields['reflectivity']['data']
    print(radar_np.shape)
    #plt.imshow(radar_np)
    #plt.show()

    
    display = pyart.graph.RadarDisplay(radar)
    fig = plt.figure(figsize=(6, 5))

    # plot super resolution reflectivityx
    ax = fig.add_subplot(111)
    display.plot('reflectivity', 0, title='NEXRAD Reflectivity',
                 vmin=-32, vmax=64, colorbar_label='', ax=ax)
    display.plot_range_ring(radar.range['data'][-1]/1000., ax=ax)
    display.set_limits(xlim=(-200, 200), ylim=(-200, 200), ax=ax)
    plt.show()
    

if __name__ == "__main__":
    main()
