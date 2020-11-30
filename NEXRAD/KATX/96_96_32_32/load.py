import cv2
import matplotlib.pyplot as plt
from os import listdir

filenames = listdir(".")
for files in filenames:
    split = files.split(".")
    if len(split) <= 1:
        continue
    ext = split[-1]
    if ext != "png":
        continue
    img = cv2.imread(files)
    cols = img.shape[1] // 2
    act = img[:, :cols, :]
    pred = img[:, cols:, :]
    cv2.imwrite("a"+files, act)
    cv2.imwrite("p"+files, pred)
