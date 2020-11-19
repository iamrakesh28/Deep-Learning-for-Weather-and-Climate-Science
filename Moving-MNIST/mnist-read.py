import matplotlib.pyplot as plt
import numpy as np
import gzip

image_rows = 28
image_cols = 28
total_images = 60000

def display(data, ind):
    #print (data.shape)
    image = np.asarray(data[ind]).squeeze()
    #print (image.shape)
    plt.imshow(image, cmap='gray')
    plt.show()
    
if __name__ == "__main__":
    num_images = 10
    fd = gzip.open('train-images-idx3-ubyte.gz','r')
    buf = fd.read(16)
    buf = fd.read(image_rows * image_cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_rows, image_cols, 1)

    display(data, 2)
