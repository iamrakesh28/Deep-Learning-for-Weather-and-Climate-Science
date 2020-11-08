import requests
import os
from lxml import html

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
    for (name, link) in zip(data_name, data_links):
        # first 23 charaters
        filename = name.lstrip()[:23]
        radar = requests.get(link)
        open(path + filename, 'wb').write(radar.content)
    

def main():
    
    url = 'https://www.ncdc.noaa.gov/nexradinv/bdp-download.jsp'
    params = {
        "id" : "PHWA",
        "yyyy" : "2020",
        "mm" : "10",
        "dd" : "31",
        "product" : "AAL2"
    }
    download_data(url, params)

if __name__ == "__main__":
    main()
