import matplotlib.pyplot as plt
import numpy as np
import cv2
import shutil
import requests
import os
from lxml import html

# from metpy.cbook import get_test_data
from metpy.io import Level2File
from metpy.plots import add_timestamp

def read_nexRad(filename):
    
    # Open the file
    # name = get_test_data('PHWA20201031_000332_V06.gz', as_file_obj=False)
    f = Level2File(filename)
    # f = filename
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
    

    fig, axes = plt.subplots(1, 1, figsize=(6, 3))
    
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
    # add_timestamp(axes, f.dt, y=0.02, high_contrast=True)
    axes.axis('off')
    # plt.show()
    
    # redraw the plot
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    width, height = fig.get_size_inches() * fig.get_dpi()
    data = np.fromstring(fig.canvas.tostring_rgb(),
                         dtype=np.uint8).reshape(int(height), int(width), 3)
    # print(data.shape)
    data = cv2.cvtColor(data[30:180, 150:300], cv2.COLOR_BGR2GRAY)
    data = cv2.resize(data, (100, 100), interpolation = cv2.INTER_AREA)
    fig.clf()
    plt.close()
    # data = cv2.blur(data, (3, 3))
    # print(data.shape)
    # plt.show()
    # plt.imshow(data, cmap='gray')
    # plt.show()
    #plt.savefig('test.png', cmap='gray')

    # save into a file
    return data

# generates radar data for a day
def save_day(dirname):
    # sort to get files in the correct sequence
    filenames = os.listdir(dirname)
    filenames.sort()

    data = []
    itern = 0
    for filename in filenames:
        path = dirname + "/" + filename
        data.append(read_nexRad(path))
        itern += 1
        if (itern % 5 == 0):
            print(itern, "files read")

    data = np.array(data)
    return data


# Downloads all the NexRad level 2 data for the day
def download_data(url, params):
    page = requests.get(url, params=params)
    tree = html.fromstring(page.content)
    data_links = tree.xpath('//div[@class="bdpLink"]/a/@href')
    data_name = tree.xpath('//div[@class="bdpLink"]/a/text()')
    print("Total NexRad level 2 data : ", len(data_links))

    # one directory for each day
    path = params["dd"] + params["mm"] + params["yyyy"] + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    # downloads the data
    cnt = 0
    filenames = set(os.listdir(path))
    for (name, link) in zip(data_name, data_links):
        cnt += 1
        if cnt % 5 == 0:
            print(cnt, "files downloaded")
        # first 23 charaters
        filename = name.lstrip()[:23]
        if filename in filenames:
            continue
        radar = requests.get(link)
        open(path + filename, 'wb').write(radar.content)
    

def main():
    
    url = 'https://www.ncdc.noaa.gov/nexradinv/bdp-download.jsp'
    params = {
        "id" : "PHWA",
        "yyyy" : "2020",
        "mm" : "10",
        "dd" : "27",
        "product" : "AAL2"
    }
    data = []
    for date in range(0, 31):
        if date < 10:
            params["dd"] = "0" + str(date)
        else:
            params["dd"] = str(date)
        download_data(url, params)
        dirname = params["mm"] + params["yyyy"]
        
        if date < 10:
            dirname = "0" + str(date) + dirname
        else:
            dirname = str(date) + dirname
            
        temp = save_day(dirname)
        if len(data) == 0:
            data = temp
        else:
            data = np.concatenate((data, temp), 0)
        #os.rmdir(dirname)
        shutil.rmtree(dirname, ignore_errors=True)
    np.save("oct", data)

if __name__ == "__main__":
    main()
