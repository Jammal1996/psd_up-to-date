# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 01:17:53 2023

@author: GPU
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

fname = r"C:\Users\GPU\particle_size_project\particle_size_distribution\thesis\plastic_mask.png"

# Reading and preporcessing the image 
img = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# # saving the image        
# cv2.imwrite(r"rgb.png", img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# saving the image        
cv2.imwrite(r"C:\Users\GPU\particle_size_project\particle_size_distribution\thesis\non-machine\plastic_non.png", img)
# # saving the image        
# cv2.imwrite(r"gray.png", img)
# Converting the image into black and white so we can detect the targeted particles 
img = cv2.threshold(img, 0, 255,
 	  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

# saving the image        
cv2.imwrite(r"C:\Users\GPU\particle_size_project\particle_size_distribution\thesis\non-machine\plastic_mask_non.png", img)

# coloring the deteceted particles 
contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img.astype(np.uint8), connectivity=4)
colors = np.random.randint(0, 255, size=(n_labels, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]  # for cosmetic reason we want the background black
false_colors = colors[labels]
cv2.imwrite(r"C:\Users\GPU\particle_size_project\particle_size_distribution\thesis\non-machine\plastic_colored_non.png", false_colors)

# Estimating the particle size distribution 
size_0_25 = []
size_25_50 = []
size_50_100 = []
size_100_250 = []
size_250_500 = []
size_500_1000 = []
size_1000_2000 = []
size_2000_4000 = []
size_4000 = []
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
areas = stats[1:, cv2.CC_STAT_AREA]
areas = list(areas)

for particle in areas:
    if particle < 25:
        size_0_25.append(particle)
    if particle >= 25 and particle < 50:
        size_25_50.append(particle)
    if particle >= 50 and particle < 100:
        size_50_100.append(particle)
    if particle >= 100 and particle < 250:
        size_100_250.append(particle)
    if particle >= 250 and particle < 500:
        size_250_500.append(particle)
    if particle >= 500 and particle < 1000:
        size_500_1000.append(particle)
    if particle >= 1000 and particle < 2000:
        size_1000_2000.append(particle)
    if particle >= 2000 and particle < 4000:
        size_2000_4000.append(particle)
    if particle > 4000:
        size_4000.append(particle)
x_axis = ['P < 25', '25 =< P < 50', '50 =< P < 100', '100 =< P < 250', '250 =< P < 500',
          '500 =< P < 1000', '1000 =< P < 2000',
          '2000 =< P < 4000', 'P > 4000']
y_axis = [len(size_0_25), len(size_25_50), len(size_50_100), len(size_100_250), len(size_250_500),
          len(size_500_1000), len(size_1000_2000), len(size_2000_4000),
          len(size_4000)]
plt.barh(x_axis, y_axis)
plt.title('Particle size P (pixels) vs num of particles')
plt.ylabel('Particle size (pixels)')
plt.xlabel('num of particles')
plt.savefig('C:/Users/GPU\particle_size_project/particle_size_distribution/thesis/non-machine/dis_plastic_non.png',dpi=400, bbox_inches = 'tight')



