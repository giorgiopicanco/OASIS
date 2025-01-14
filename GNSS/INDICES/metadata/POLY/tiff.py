#!/usr/bin/env python3
import numpy as np
import sys
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.colors as colors

# criar arrays de latitude e longitude
lats = np.arange(-90, 90, 0.5)
lons = np.arange(-180, 180, 0.5)

# criar matriz de zeros
matrix = np.zeros((lats.size, lons.size))
matrix1 = np.zeros((lats.size, lons.size))

# criar mapa de base
m = Basemap(projection='cyl', llcrnrlon=lons[0], llcrnrlat=lats[0], urcrnrlon=lons[-1], urcrnrlat=lats[-1])


# preencher a matriz1 com 1 para as regiões continentais excluindo a África
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        x, y = m(lon, lat)
        if m.is_land(x, y):
            matrix1[i, j] = 1

# preencher a matriz com 1 para as regiões continentais e as regiões próximas ao continente
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        x, y = m(lon, lat)
        if m.is_land(x, y) or lat < -60:  # Considerar a Antártica (lat < -60)
            matrix[i, j] = 1

# criar máscara com expansão de 1000 km a partir das regiões continentais
mask = morphology.binary_dilation(matrix, iterations=20) # 20 = 1000km

# preencher a matriz com 1 para as regiões continentais e as regiões próximas ao continente
matrix[mask == 1] = 1

plt.rcParams['font.family'] = 'Palatino Linotype'
# plotar os mapas lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Definir cores personalizadas
colors_custom = ['blue', 'yellow']

# Criar colormap personalizado
cmap1 = colors.ListedColormap(colors_custom)

# plotar o primeiro mapa à esquerda

im1 = ax1.imshow(matrix1, origin='lower', extent=[lons[0], lons[-1], lats[0], lats[-1]], cmap=cmap1)
ax1.set_xlabel('Longitude (°)',fontsize=14)
ax1.set_ylabel('Latitude (°)',fontsize=14)
ax1.set_title('Região Continental',fontsize=14)
cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1], shrink=0.5)
cbar1.ax.set_ylabel('Domínio', rotation=90, fontsize=14, labelpad=10)  # Posicionar o título ao lado
cbar1.ax.tick_params(labelsize=16)
cbar1.ax.yaxis.set_tick_params(width=0)
ax1.set_yticks(np.arange(-90, 91, 20))

# plotar o segundo mapa à direita
im2 = ax2.imshow(matrix, origin='lower', extent=[lons[0], lons[-1], lats[0], lats[-1]], cmap=cmap1)
ax2.set_xlabel('Longitude (°)',fontsize=14)
ax2.set_ylabel('Latitude (°)',fontsize=14)
ax2.set_title('Região Continental com extensão de 500 km',fontsize=14)
cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1], shrink=0.5)
ax2.set_yticks(np.arange(-90, 91, 20))

# Ajustar o tamanho do título da barra de cores
cbar2.ax.set_ylabel('Domínio', rotation=90, fontsize=14, labelpad=10)  # Posicionar o título ao lado


cbar2.ax.tick_params(labelsize=14)
cbar2.ax.yaxis.set_tick_params(width=0)

# ajustar o espaçamento entre os subplots
#plt.tight_layout()

# ajustar o espaçamento entre os subplots
plt.subplots_adjust(wspace=0.4)

# salvar a figura com o mesmo nome do arquivo de texto
fig.savefig('new_matriz_binaria_500km.png', dpi=300)


# exportar a matriz para um arquivo de texto separado por tabulações
np.savetxt('mapa_continente.txt', matrix, delimiter=' ', fmt='%d')
print("Matriz exportada para 'mapa_continente.txt'.")

