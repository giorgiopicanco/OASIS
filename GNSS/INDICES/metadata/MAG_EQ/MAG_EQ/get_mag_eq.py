#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pyIGRF
import sys
import numpy as np
import re
from numpy import ndarray
import datetime
import os

def decimal_year(day, month, year):
    d = datetime.date(year, month, day)
    total_days_in_year = 365 if not d.year % 4 == 0 else 366
    day_of_year = d.timetuple().tm_yday
    return year + (day_of_year - 1) / total_days_in_year


YEARS = np.arange(2024, 2025, 1)
MONTHS = np.arange(5, 12, 1)

for yr in YEARS:
    y = yr
    for m in range(len(MONTHS)):
        m=(MONTHS[m])

        #Data para calculo das coordenadas do equador magnetico:
        
        m = m
        d = 15
        

        data=(decimal_year(d, m, y))
        #print(data)

        #Latitudes e Longitudes
        step = 1 #graus
        latitudes = np.arange(-20, 21, step)
        longitudes = np.arange(0, 361, step)

        lat2 = []

        for i in range(len(latitudes)):
            lat=latitudes[i]
            for i in range(len(longitudes)):
                lon = longitudes[i]
                alt = 1000.0  # altitude em metros

                # Calcula os valores do campo magnético
                D, I, H, X, Y, Z, F = pyIGRF.igrf_value(lat, lon, alt, data)
                xyz=(lat, lon, I)
                lat2.append(xyz)

        # Tupla de 3 colunas separadas por vírgulas
        tupla = lat2

        # Inicializa os arrays individuais
        col1 = []
        col2 = []
        col3 = []

        # Separa a tupla em arrays individuais
        for linha in tupla:
            col1.append(linha[0])
            col2.append(linha[1])
            col3.append(linha[2])

        latitudes = np.array(col1)
        longitudes = np.array(col2)
        longitudes = (longitudes + 180) % 360 - 180
        declinacao = np.array(col3)

        #Define o range de inclinacao magnetica para o equador
        deg1 = -0.5
        deg2 = 0.5
        indices = np.where((declinacao >= deg1) & (declinacao <= deg2))

        latitudes = latitudes[indices]
        longitudes = longitudes[indices]
        declinacao = declinacao[indices]




        parent_dir = str(y)
        month1='{:02d}'.format(m)
        
        month_dir = os.path.join(parent_dir, str(month1))
        
        print(month_dir)
                   
        if not os.path.exists(month_dir):
            os.makedirs(month_dir)
                    
        print(f"Calculating magnetic equator for {y}-{month1:02}")            
        filename = os.path.join(month_dir, "{}_{}.EQ".format(y, month1))
      

        with open(filename, 'w') as f:
            for i in range(len(longitudes)):
                line=str("%13.2f" % (longitudes[i]))+' '+str("%13.2f" % (latitudes[i]))
                f.write(line)
                f.write('\n')
            if f:
                f.close()


        #plt.scatter(longitudes,latitudes,color='red')
        #plt.xlim(-180, 180)
        #plt.ylim(-90, 90)
        #plt.yticks([-90, -60, -30, 0, 30, 60, 90])
        #plt.xticks([-180, -120, -60, 0, 60, 120, 180])
        #plt.savefig("teste.png")
        #plt.show()

