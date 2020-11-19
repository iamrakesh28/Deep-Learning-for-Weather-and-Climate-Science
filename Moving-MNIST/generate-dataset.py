import matplotlib.pyplot as plt
import numpy as np
import random
import math
import gzip

image_rows = 28
image_cols = 28
total_images = 60000

total_test = 1000
frame_no = 20
# (row, col)
frame_size = (64, 64)
top_left = (13, 13)

# image shouldn't go outside
def valid(r, c):
    if r >= 0 and r < frame_size[0] and c >= 0 and c < frame_size[1]:
        return True
    return False

'''
def move_corner(corner_):
    # constant appilcable for this image type
    success = False
    corner = None
    while (success == False):
        dr = random.randint(-2, 2)
        dc = random.randint(-2, 2)
        corner = (corner_[0] + dr, corner_[1] + dc)
        horz = (corner[1], corner[1] + image_cols - 1)
        vert = (corner[0], corner[1] + image_rows - 1)
        #print (center, horz, vert)
        success = valid(horz, vert)
    return (dr, dc)
'''

# 3 images    
def cons_frame(imgs, corners, vel):
    
    frame = np.zeros(frame_size, dtype=np.uint8)
    
    for i in range(len(imgs)):
        for r in range(imgs[i].shape[0]):
            for c in range(imgs[i].shape[1]):
                x, y = (r + vel[i][0] + corners[i][0], c + vel[i][1] + corners[i][1])

                if valid(x, y) == False:
                    rev_x = 1
                    rev_y = 1
                    if valid(x, 0) == False:
                        rev_x = -1
                    if valid(0, y) == False:
                        rev_y = -1
                        
                    vel[i] = (rev_x * vel[i][0], rev_y * vel[i][1])
                    x, y = (r + vel[i][0] + corners[i][0], c + vel[i][1] + corners[i][1])
                    
                frame[x][y] = max(frame[x][y], imgs[i][r][c])

    corners = ((corners[0][0] + vel[0][0], corners[0][1] + vel[0][1]),
               (corners[1][0] + vel[1][0], corners[1][1] + vel[1][1])
    )
    return (frame, corners, vel)

def gen_test(data, num_images):
    choices = ()
    corners = ()
    velocity = []
    count = 0

    # two images per frame
    while count < 2:
        choices += (random.randint(0, num_images - 1), )
        count += 1

    count = 0
    while count < 2:
        r = random.randint(0, frame_size[0] - 28 - 1)
        c = random.randint(0, frame_size[1] - 28 - 1)
        corners += ((r, c), )
        count += 1

    count = 0
    while count < 2:
        amp = random.randint(3, 5)
        theta = random.uniform(0, 2 * math.pi)
        vr = int(amp * math.cos(theta))
        vc = int(amp * math.sin(theta))
        velocity += ((vr, vc), )
        count += 1

    imgs = ()
    for ind in choices:
        #display(data, ind)
        imgs += (np.asarray(data[ind]).squeeze(), )

    frames = []
    for i in range(frame_no):
        frame, corners, velocity = cons_frame(imgs, corners, velocity)
        frames.append(frame)
        #plt.imshow(frame, cmap='gray')
        #plt.show()
    return frames

def dataset(data, num_images):
    train = []
    #for i in range(total_test):
    #    train.append(gen_test(data, num_images))
    #print(train)
    #np.save('train' + str(total_test) + '.npy', train)
    train = np.load('train' + str(total_test) + '.npy')
    print (train.shape)
    plt.imshow(train[0][2], cmap='gray')
    plt.show()

def display(data, ind):
    #print (data.shape)
    image = np.asarray(data[ind]).squeeze()
    #print (image.shape)
    plt.imshow(image, cmap='gray')
    plt.show()
    
if __name__ == "__main__":
    num_images = 500
    fd = gzip.open('train-images-idx3-ubyte.gz','r')
    buf = fd.read(16)
    buf = fd.read(image_rows * image_cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_rows, image_cols, 1)

    dataset(data, num_images)
    #print (move_corner(top_left))
    #display(data, 2)

