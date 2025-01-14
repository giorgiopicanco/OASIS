#!/usr/bin/env python3

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from shapely.geometry import MultiPoint
import sys
import os

# Carregar o arquivo shapefile
gdf = gpd.read_file('/home/debian-giorgio/OASIS/INDICES/metadata/POLY/América/América.shp')

# Filtrar apenas os países desejados
paises_desejados = ['Brasil', 'Argentina', 'México', 'Bolivia', 'Chile', 'Perú',
                    'Paraguay', 'Colombia', 'Costa Rica', 'Ecuador', 'Guayana Francesa (Francia)', 'Uruguay',
                    'Venezuela', 'Surinam', 'Guyana', 'Panamá', 'Nicaragua', 'El Salvador', 'Honduras',
                    'Guatemala', 'Belice']
gdf = gdf[gdf['PAÍS'].isin(paises_desejados)]

# Concatena as geometrias dos países em um único GeoDataFrame
paises = gpd.GeoDataFrame(pd.concat([gdf[gdf['PAÍS']==pais] for pais in paises_desejados], ignore_index=True), crs=gdf.crs)

# Une todas as geometrias em uma única geometria
geometria_unida = unary_union(paises.geometry)

# Obtém apenas as bordas da geometria
bordas = gpd.GeoDataFrame(geometry=[geometria_unida.boundary], crs=paises.crs)

# # Plota as bordas
# fig, ax = plt.subplots(figsize=(10,10))
# bordas.plot(ax=ax, color='green')
# plt.show()



# # Cria o diretório "Bordas" se ele não existir
# if not os.path.exists('Bordas'):
    # os.makedirs('Bordas')

# # Salva o shapefile dentro do diretório "Bordas"
# bordas.to_file('Bordas/bordas.shp')




import numpy as np
import matplotlib.pyplot as plt
import sys
import geopandas as gpd
from shapely.geometry import Point, box

# carregar shapefile
bordas = gpd.read_file("'/home/debian-giorgio/OASIS/INDICES/metadata/POLY/Bordas/bordas.shp")


# criar arrays de latitude e longitude
lats = np.arange(-80, 40, 0.5)
lons = np.arange(-110, 0, 0.5)

# Lendo a matriz a partir do arquivo de texto
with open("DIXMAP_2015_144_0250.MAP") as file:
    # Lendo todas as linhas do arquivo
    lines = file.readlines()

    # Criando uma lista com os valores da matriz
    values = []
    for line in lines:
        # Dividindo a linha em valores separados por espaços
        row = line.strip().split()
        # Convertendo os valores para float
        row = [float(val) if val != '-999.0' else np.nan for val in row]
        # Adicionando a linha à lista de valores
        values.append(row)

# Convertendo a lista de valores para um array numpy
tec = np.array(values)

# criar um array para guardar os pontos
points = []

# criar arrays para guardar as posições das linhas e colunas
rows = np.arange(0, 240, 1)
cols = np.arange(0, 220, 1)






# definir os limites da área de interesse
xmin, ymin, xmax, ymax = [-110, -80, 0, 40]

# criar uma camada de polígonos que cobre toda a área de interesse
grid = gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs=bordas.crs)

# sobrepor as camadas
intersection = gpd.overlay(bordas, grid, how='intersection')


# plotar mapa
fig, ax = plt.subplots(figsize=(10,10))
plt.imshow(tec, extent=[-110, 0, -80, 40], cmap='jet', vmin=0, vmax=10)
bordas.plot(ax=ax, color='black')
plt.colorbar()
plt.show()


