# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:50:35 2023

@author: GPU
"""

import cv2
import matplotlib.pyplot as plt

fname = r"C:\Users\GPU\particle_size_project\particle_size_distribution\real-samples\wood\real-wood-1.jpg"
img = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
img = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY_INV)[1]
# print(img)
cv2.imwrite("img.jpg", img)

size_0_100 = []
size_100_250 = []
size_250_500 = []
size_500_1000 = []
size_1000_2000 = []
size_2000_4000 = []
size_4000 = []
        
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
areas = stats[1:, cv2.CC_STAT_AREA]
areas = list(areas)
print(len(areas))
for particle in areas:
    if particle < 100:
        size_0_100.append(particle)
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
x_axis = ['P < 100', '100 =< P < 250', '250 =< P < 500',
          '500 =< P < 1000', '1000 =< P < 2000',
          '2000 =< P < 4000', 'P > 4000']
y_axis = [len(size_0_100), len(size_100_250), len(size_250_500),
          len(size_500_1000), len(size_1000_2000), len(size_2000_4000),
          len(size_4000)]
plt.barh(x_axis, y_axis)
plt.title('Particle size P (pixels) vs num of particles')
plt.ylabel('Particle size (pixels)')
plt.xlabel('num of particles')
plt.savefig('dist.jpeg')
