import matplotlib.pyplot as plt
import numpy as np

from metpy.cbook import get_test_data
from metpy.io import Level2File
from metpy.plots import add_timestamp

def read_nexRad(filename):
    
    # Open the file
    # name = get_test_data('PHWA20201031_000332_V06.gz', as_file_obj=False)
    f = Level2File(filename)
    
    # print(f.sweeps[0][0])
    # Pull data out of the file
    sweep = 0
    
    # First item in ray is header, which has azimuth angle
    az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])

    # 5th item is a dict mapping a var name (byte string) to a tuple
    # of (header, data array)
    ref_hdr = f.sweeps[sweep][0][4][b'REF'][0]
    ref_range = np.arange(ref_hdr.num_gates) * ref_hdr.gate_width + ref_hdr.first_gate
    ref = np.array([ray[4][b'REF'][1] for ray in f.sweeps[sweep]])
    
    # rho_hdr = f.sweeps[sweep][0][4][b'RHO'][0]
    # rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
    # rho = np.array([ray[4][b'RHO'][1] for ray in f.sweeps[sweep]])
    

    fig, axes = plt.subplots(1, 1, figsize=(15, 8))
    
    # reflexivity plot
    data = np.ma.array(ref)
    data[np.isnan(data)] = np.ma.masked
    
    # Convert az,range to x,y
    xlocs = ref_range * np.sin(np.deg2rad(az[:, np.newaxis]))
    ylocs = ref_range * np.cos(np.deg2rad(az[:, np.newaxis]))

    # Plot the data
    axes.pcolormesh(xlocs, ylocs, data, cmap='viridis')
    axes.set_aspect('equal', 'datalim')
    axes.set_xlim(-150, 150)
    axes.set_ylim(-150, 150)
    #plt.axis('off')
    plt.show()
    return
    # redraw the plot
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    width, height = fig.get_size_inches() * fig.get_dpi()
    data = np.fromstring(fig.canvas.tostring_rgb(),
                         dtype=np.uint8).reshape(int(height), int(width), 3)
    print(data.shape)
    plt.show()
    plt.imshow(data)
    plt.savefig('test.png', cmap='gray')
    
    '''
    
    Example code for plotting ref and rho

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    add_metpy_logo(fig, 190, 85, size='large')
    for var_data, var_range, ax in zip((ref, rho), (ref_range, rho_range), axes):
        # Turn into an array, then mask
        data = np.ma.array(var_data)
        data[np.isnan(data)] = np.ma.masked
        
        # Convert az,range to x,y
        xlocs = var_range * np.sin(np.deg2rad(az[:, np.newaxis]))
        ylocs = var_range * np.cos(np.deg2rad(az[:, np.newaxis]))
        print((xlocs.shape), (ylocs.shape), (data.shape))
        # Plot the data
        ax.pcolormesh(xlocs, ylocs, data, cmap='viridis')
        ax.set_aspect('equal', 'datalim')
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        plt.savefig('test.png')
        #add_timestamp(ax, f.dt, y=0.02, high_contrast=True)
    

    plt.show()
    '''

def main():
    filename = "KATX20200130_001651_V06"
    read_nexRad(filename)

if __name__ == "__main__":
    main()
