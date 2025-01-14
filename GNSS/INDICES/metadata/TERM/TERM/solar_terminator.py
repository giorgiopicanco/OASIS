#!/usr/bin/env python3

import sys
import math
import ephem
import numpy as np
import matplotlib.pyplot as plt

# y = int(str(sys.argv[1]))
# m = int(str(sys.argv[2]))
# d = int(str(sys.argv[3]))
# h = int(str(sys.argv[4]))
# mi = int(str(sys.argv[5]))
# s = int(str(sys.argv[6]))


months = np.arange(1, 13, 1)
hours = np.arange(0, 24, 1)
minutes = np.arange(0, 60, 10)

#print(months)
#print(hours)
#print(minutes)

#sys.exit()
y = 2018
m = 5
d = 18
#h = 12
mi = 0
s = 0

lon1=[]
lat1=[]
h1=[]
m1=[]
for i in range(len(hours)):
    h=(hours[i])
    print(h)
    for j in range(len(minutes)):
        print(minutes[j])
        m=minutes[j]
            

        sun = ephem.Sun((y,m,d,h,mi,s))

        sun.compute((y,m,d,h,mi,s))

        #print sun.ra,sun.dec

        greenwich = ephem.Observer()
        greenwich.date = (y,m,d,h,mi,s)

        #print(greenwich.sidereal_time())

        lon = math.degrees(-greenwich.sidereal_time() + sun.ra)
        if lon > 180:
            lon = lon - 360

        lat = math.degrees(sun.dec)

        print(lon,lat)
        lon1.append(lon)
        lat1.append(lat)
        h1.append(h)
        m1.append(m)


for i in range(len(h1)):
    print(h1[i],m1[i],lon1[i],lat1[i])



