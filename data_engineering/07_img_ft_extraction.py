
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import os

path = "06_satellites images/images_for_every_block"

files = os.listdir(path)

def green_vectorized(img):
    mask = (img[:,:,1] > img[:,:,0]) & (img[:,:,1] > img[:,:,2]) & ((img[:,:,1]/np.sum(img, axis=2)) > .35)
    return round(100 * np.sum(mask)/(img.shape[0]*img.shape[1]), 4)

def get_textural_features(img):
    green = green_vectorized(img)
    img = img_as_ubyte(rgb2gray(img))
    glcm = graycomatrix(img, [1], [0], 256, symmetric=True, normed=True)
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    feature = [dissimilarity, correlation, homogeneity, energy, green]
    return feature

import time
import pandas as pd
db = []
t1=time.time()
for j in files:
    img = io.imread(f"{path}/{j}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pack = get_textural_features(img_rgb)
    pack.append(j.replace("pic_","").replace(".jpg",""))
    db.append(pack)
t2=time.time()

total=t2-t1

cols = ['dissimilarity', 'correlation', 'homogeneity', 'energy', 'green', 'manzana']
df = pd.DataFrame(db, columns = cols)
df.to_excel("10_feature_extraction/features_of_img_blocks.xlsx", index = False)
