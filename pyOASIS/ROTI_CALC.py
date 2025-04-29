#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import numpy.ma as ma
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from pyOASIS import gnss_freqs

# # Definindo fonte Palatino Linotype em tudo
# plt.rcParams['font.family'] = 'Palatino'


def ROTIcalc(estacao,doy,ano,diretorio_principal,destination_directory):

    # Arguments from the management program
    # estacao = sys.argv[1]
    # doy = sys.argv[2]
    # ano = sys.argv[3]
    # diretorio_principal = sys.argv[4]
    # destination_directory = sys.argv[5]
    
    # Variables and Parameters
    h1 = 0
    n_horas = 24  # horas
    int1 = 320  # minutos
    
    # Accessing the frequencies of the GPS system
    gps_freqs = gnss_freqs.FREQUENCY[gnss_freqs.GPS]
    f1 = gps_freqs[1]
    f2 = gps_freqs[2]
    f5 = gps_freqs[5]
    
    akl = 40.3 * 10 ** 16 * ((1 / f2 ** 2) - (1 / f1 ** 2))
    akl15 = 40.3 * 10 ** 16 * ((1 / f5 ** 2) - (1 / f1 ** 2))
    
    # Adjustable window to calculate averages, deviations, etc., in seconds...
    # Defines the sampling frequency of the calculated index...
    # Here it is defined as 2.5 minutes (following Giorgio's method)...
    WINDOW = 60.0 * 2.5
    
    # estándar de unidad de tiempo para que el ROTI quede en TECU/min
    TROTI = 60.0
    
    # faltante de datos por pasadas sucesivas o por cortes prolongados debido a obstrucciones, etc...
    GAP = 1.5 * WINDOW
    
    # altura de la capa
    H = 350000  # m
    
    # tolerancia para la detección de outliers...
    SIGMA = 5
    
    # grado de los polinomios
    DE = 3
    
    # separación entre arcos (i.e., pasadas)
    GAP2 = 3600  # s
    
    # anglo de elevacion (cutoff):
    elev_angle=30
    
    
    # Construindo o caminho completo para a pasta CHPI
    caminho_ = os.path.join(diretorio_principal, estacao)
    print(caminho_)
    # Verificando se o diretório CHPI existe
    if os.path.exists(caminho_):
        # Listando o conteúdo da pasta CHPI
        conteudo_ = os.listdir(caminho_)
        print("Arquivos .txt na pasta:")
    
        # Definindo a variável arquivos de acordo com o conteúdo_chpi
        arquivos = [arquivo for arquivo in conteudo_ if arquivo.startswith(estacao) and arquivo.endswith(".RNX3")]
        print(arquivos)
    
    
    
        # Ordenando os arquivos pelo número do satélite
        arquivos_ordenados = sorted(arquivos, key=lambda x: int(x.split("_")[1][1:]))
    
        # Imprimindo os arquivos encontrados
        for arquivo in arquivos_ordenados:
            print("Conteúdo do arquivo", arquivo)
    
    
        print()
        # Contando o número de arquivos
        numero_de_arquivos = len(arquivos_ordenados)
        print("Número de arquivos RINEX_SCREENED no diretório:", numero_de_arquivos)
    
    else:
        print("O diretório especificado não existe.")
    
    
    print()
    
    # Inicializando listas para armazenar os valores de cada variável de todos os arquivos
    # timestamp = []
    # mjd = []
    # position = []
    # LGF = []
    # satellites = []
    # sta = []
    # obs_La = []
    # obs_Lb = []
    # obs_Ca = []
    # obs_Cb = []
    
    
    date = []
    time2 = []
    mjd = []
    pos_x = []
    pos_y = []
    pos_z = []
    LGF_combination = []
    LGF_combination15 = []
    satellites = []
    sta = []
    hght = []
    el = []
    lonn = []
    latt = []
    obs_La = []
    obs_Lb = []
    obs_Lc = []
    obs_Ca = []
    obs_Cb = []
    obs_Cc = []
    
    for arquivo in arquivos_ordenados:
        caminho_arquivo = os.path.join(caminho_, arquivo)
    
        with open(caminho_arquivo, 'r') as f:
            # Lendo o cabeçalho do arquivo
            header = f.readline().strip().split('\t')
    
            # Lendo cada linha de dados do arquivo
            for linha in f:
                # Dividindo a linha em colunas
                colunas = linha.strip().split('\t')  # Supondo que as colunas estejam separadas por tabulação (\t)
                # Associando cada coluna com o cabeçalho correspondente
                registro = {
                    'date': colunas[0],
                    'time2': colunas[1],
                    'mjd': colunas[2],
                    'pos_x': colunas[3],
                    'pos_y': colunas[4],
                    'pos_z': colunas[5],
                    'LGF_combination': colunas[6],
                    'LGF_combination15': colunas[7],
                    'satellite': colunas[8],
                    'sta': colunas[9],
                    'hght': colunas[10],
                    'el': colunas[11],
                    'lonn': colunas[12],
                    'latt': colunas[13],
                    'obs_La': colunas[14],
                    'obs_Lb': colunas[15],
                    'obs_Lc': colunas[16],
                    'obs_Ca': colunas[17],
                    'obs_Cb': colunas[18],
                    'obs_Cc': colunas[19]
                }
    
                # Adicionando os valores de cada variável às respectivas listas
                # timestamp.append(registro['timestamp'])
                date.append(registro['date'])
                time2.append(registro['time2'])
                mjd.append(registro['mjd'])
                pos_x.append(registro['pos_x'])
                pos_y.append(registro['pos_y'])
                pos_z.append(registro['pos_z'])
                LGF_combination.append(registro['LGF_combination'])
                LGF_combination15.append(registro['LGF_combination15'])
                satellites.append(registro['satellite'])
                sta.append(registro['sta'])
                hght.append(registro['hght'])
                el.append(registro['el'])
                lonn.append(registro['lonn'])
                latt.append(registro['latt'])
                obs_La.append(registro['obs_La'])
                obs_Lb.append(registro['obs_Lb'])
                obs_Lc.append(registro['obs_Lc'])
                obs_Ca.append(registro['obs_Ca'])
                obs_Cb.append(registro['obs_Cb'])
                obs_Cc.append(registro['obs_Cc'])
    
    # Criar uma única figura
    plt.figure(figsize=(12, 6))
    
    
    # Definindo a paleta de cores
    palette = plt.get_cmap('tab10')
    
    #sat = 'R'
    sat_classes = ['G','R']
    
    for sat in sat_classes:
        satx=sat
        # Filtrando os satélites
        if satx:
            satellites_to_plot = [sv for sv in np.unique(satellites) if sv.startswith(sat)]
        else:
            satellites_to_plot = np.unique(satellites)
    
        print(satellites_to_plot)
    
    
        # Criar uma única figura
        # plt.figure(figsize=(12, 6))
    
        # Inicializando uma lista para armazenar os valores de ROTI de todos os satélites
        # Lista para armazenar os dicionários de dados de cada satélite
        dados_satelites = []
    
        for sat1 in satellites_to_plot:
            print(sat1)
            indices = np.where(np.array(satellites) == sat1)[0]
            # print(f"Índices para o satélite {satellite}")#  : {indices}")
    
            # Inicializando listas filtradas para cada satélite
            date_filtered = []
            time2_filtered = []
            mjd_filtered = []
            pos_x_filtered = []
            pos_y_filtered = []
            pos_z_filtered = []
            LGF_combination_filtered = []
            LGF_combination15_filtered = []
            satellites_list_filtered = []
            sta_filtered = []
            hght_filtered = []
            el_filtered = []
            lonn_filtered = []
            latt_filtered = []
            obs_La_filtered = []
            obs_Lb_filtered = []
            obs_Lc_filtered = []
            obs_Ca_filtered = []
            obs_Cb_filtered = []
            obs_Cc_filtered = []
    
            for idx in indices:
                date_filtered.append(date[idx])
                time2_filtered.append(time2[idx])
                mjd_filtered.append(mjd[idx])
                pos_x_filtered.append(pos_x[idx])
                pos_y_filtered.append(pos_y[idx])
                pos_z_filtered.append(pos_z[idx])
                LGF_combination_filtered.append(LGF_combination[idx])
                LGF_combination15_filtered.append(LGF_combination15[idx])
                satellites_list_filtered.append(satellites[idx])
                sta_filtered.append(sta[idx])
                hght_filtered.append(hght[idx])
                el_filtered.append(el[idx])
                lonn_filtered.append(lonn[idx])
                latt_filtered.append(latt[idx])
                obs_La_filtered.append(obs_La[idx])
                obs_Lb_filtered.append(obs_Lb[idx])
                obs_Lc_filtered.append(obs_Lc[idx])
                obs_Ca_filtered.append(obs_Ca[idx])
                obs_Cb_filtered.append(obs_Cb[idx])
                obs_Cc_filtered.append(obs_Cc[idx])
    
            data = {
                'date': date_filtered,
                'time': time2_filtered,
                'mjd': mjd_filtered,
                'pos_x': pos_x_filtered,
                'pos_y': pos_y_filtered,
                'pos_z': pos_z_filtered,
                'LGF': LGF_combination_filtered,
                'LGF15': LGF_combination15_filtered,
                'satellites': satellites_list_filtered,
                'sta': sta_filtered,
                'hh': hght_filtered,
                'elev': el_filtered,
                'lonn': lonn_filtered,
                'latt': latt_filtered,
                'obs_La': obs_La_filtered,
                'obs_Lb': obs_Lb_filtered,
                'obs_Lc': obs_Lc_filtered,
                'obs_Ca': obs_Ca_filtered,
                'obs_Cb': obs_Cb_filtered,
                'obs_Cc': obs_Cc_filtered
            }
    
    
            df = pd.DataFrame(data)
    
    
    
            # Convertendo as colunas 'date' e 'time' para o tipo datetime e depois concatenando
            df['timestamp'] = df['date'] + ' ' + df['time']
    
            # Convertendo a coluna 'timestamp' para o tipo datetime, se ainda não estiver
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    
            # Convertendo as colunas relevantes para float
            columns_to_convert = ['LGF', 'LGF15', 'mjd','lonn','latt','hh','elev']
            df[columns_to_convert] = df[columns_to_convert].astype(float)
    
            df.replace(-999999.999, np.nan, inplace=True)
    
    
            # L1-L2
    
    
    
            t = df['mjd']
            stec = df['LGF'] / akl
            stec15 = df['LGF15'] / akl15
            lat = df['latt']
            lon = df['lonn']
            elev = df['elev']
            hh = df['hh']
    
    
    
            # diferencia entre épocas consecutivas...
            d = 86400.0*np.diff(t)
    
            # crear máscara porque en la frontera de los 15 minutos aparecen épocas repetidas...
            d = ma.masked_values(d,0.0)
    
            # buscar y corregir "saltos" de IFBs residuales (entre ajustes)...
            i = np.where(np.append(d.mask, False) == True)[0]
            for j in range(i.size):
                dIFB = stec[i[j] + 1] - stec[i[j]]
                stec[i[j] + 1:] = stec[i[j] + 1:] - dIFB
    
                dIFB15 = stec15[i[j] + 1] - stec15[i[j]]
                stec15[i[j] + 1:] = stec15[i[j] + 1:] - dIFB15
    
            # no se requiere aquí...
            #RATE = np.round(np.min(x[~ x.mask]))
    
            # si al menos un dato debió ser enmascarado (i.e., al menos una época repetida)...
            if (d.mask.any()):
                lat = lat[np.append(~ d.mask, True)]
                lon = lon[np.append(~ d.mask, True)]
                hh = hh[np.append(~ d.mask, True)]
                # az = az[np.append(~ d.mask, True)]
                t = t[np.append(~ d.mask, True)]
                stec = stec[np.append(~ d.mask, True)]
                stec15 = stec15[np.append(~ d.mask, True)]
                elev = elev[np.append(~ d.mask, True)]
    
    
            # primer observación...
            t0 = t[0]
    
            # índice de datos en cada ventana...
            i = np.floor(np.round(86400*(t - t0))/WINDOW)
    
            # ventanas con datos...
            j = np.unique(i)
    
    
            # acumuladores...
            alon = []
            alat = []
            at = []
            ahh = []
            # aaz = []
            ROTI = []
            ROTI15 = []
            elev1 = []
    
    
    
    
            # acá calculo el ROTI (a modo de ejemplo, aunque debería ser con STEC y no con stec)...
            for k in range(j.size):
                l = ma.masked_values(i,j[k])
    
                # solo si hay al menos de 2 observaciones en esta ventana...
                if lon[l.mask].size > 1:
                    alon.append(np.mean(lon[l.mask]))
                    alat.append(np.mean(lat[l.mask]))
                    ahh.append(np.mean(hh[l.mask]))
                    # aaz.append(np.mean(az[l.mask]))
                    at.append(np.mean(t[l.mask]))
                    elev1.append(np.mean(elev[l.mask]))
    
                    ROT = np.divide(np.diff(stec[l.mask]),86400.0*np.diff(t[l.mask])/TROTI)
    
                    ROT15 = np.divide(np.diff(stec15[l.mask]),86400.0*np.diff(t[l.mask])/TROTI)
    
                    # el np.abs() "captura" error de redondeo cuando el ROTI es muy muy muy muy pequeño, casi cero (i.e., 1e-19)...
                    ROTI.append(np.sqrt(np.abs(np.mean(ROT*ROT) - np.mean(ROT)**2)))
    
                    ROTI15.append(np.sqrt(np.abs(np.mean(ROT15*ROT15) - np.mean(ROT15)**2)))
    
    
    
            ## matrices..
            alon = np.array(alon)
            alat = np.array(alat)
            ahh = np.array(ahh)
            # aaz = np.array(aaz)
            at = np.array(at)
            ROTI = np.array(ROTI)
            ROTI15 = np.array(ROTI15)
            elev = np.array(elev1)
    
    
            # diferencia entre épocas consecutivas...
            d = 86400.0*np.diff(at)
    
            # máscara para buscar arcos independientes...
            d = ma.masked_greater_equal(d,GAP2)
    
            # donde ocurren los gaps de tiempo?
            i = np.where(np.append(d.mask, False) == True)[0]
    
            # donde empiezan los arcos?
            i1 = np.append(0,np.where(np.append(d.mask, False) == True)[0])
            # donde terminan los arcos?
            i2 = np.append(np.where(np.append(d.mask, False) == True)[0] - 1,alon.size)
    
            # matrices para los polinomios ajustados y los límites superior e inferior...
            y = np.empty((ROTI.size,))
            yup = np.empty((ROTI.size,))
            ydown = np.empty((ROTI.size,))
    
            y15 = np.empty((ROTI15.size,))
            yup15 = np.empty((ROTI15.size,))
            ydown15 = np.empty((ROTI15.size,))
    
            #filtrado por "spikes" debido a los diferentes "intervalos" nivelados con ambigüedad constante, cuyos límites no son aun expuestos por AGEO...
            for j in range(i1.size):
                # época media
                tm = np.mean(at[i1[j]:i2[j]])
    
                if (at[i1[j]:i2[j]][-1] - at[i1[j]:i2[j]][0]) != 0.0:
                    # variable independiente normalizada
                    x = (at[i1[j]:i2[j]] - tm)/(at[i1[j]:i2[j]][-1] - at[i1[j]:i2[j]][0])
    
                    # ajuste del polinomio
                    c = np.polyfit(x,ROTI[i1[j]:i2[j]],DE)
    
                    c15 = np.polyfit(x,ROTI15[i1[j]:i2[j]],DE)
    
                    # evaluación del polinomio
                    y[i1[j]:i2[j]] = np.polyval(c,x)
    
                    y15[i1[j]:i2[j]] = np.polyval(c15,x)
    
                    # rms de los residuos
                    rms = np.std(ROTI[i1[j]:i2[j]] - y[i1[j]:i2[j]])
    
                    rms15 = np.std(ROTI15[i1[j]:i2[j]] - y15[i1[j]:i2[j]])
                else:
                    y[i1[j]:i2[j]] = ROTI[i1[j]:i2[j]]
                    rms = 0.0
    
                    y15[i1[j]:i2[j]] = ROTI15[i1[j]:i2[j]]
                    rms15 = 0.0
    
                # límites superior e inferior
                yup[i1[j]:i2[j]] = y[i1[j]:i2[j]] + SIGMA*rms
                ydown[i1[j]:i2[j]] = y[i1[j]:i2[j]] - SIGMA*rms
    
                yup15[i1[j]:i2[j]] = y15[i1[j]:i2[j]] + SIGMA*rms15
                ydown15[i1[j]:i2[j]] = y15[i1[j]:i2[j]] - SIGMA*rms15
    
    
            # hacer máscara
            mask = np.abs(ROTI - y) > (yup - ydown)/2.0
    
            # descartar los valores enmascarados (los outliers)...
            alatm = alat[~ mask]
            alonm = alon[~ mask]
            ahhm = ahh[~ mask]
            # aazm = aaz[~ mask]
            atm = at[~ mask]
            ROTIm = ROTI[~ mask]
            ROTIm15 = ROTI15[~ mask]
            elevm = elev[~ mask]
    
    
            cutoff = np.where(elevm>=elev_angle)
    
    
    
            alat = alatm[cutoff]
            alon = alonm[cutoff]
            ahh = ahhm[cutoff]
            # aaz = aazm[cutoff]
            at = atm[cutoff]
            ROTI = ROTIm[cutoff]
            ROTI15 = ROTIm15[cutoff]
            elev = elevm[cutoff]
    
    
            cut_out = np.where(ROTI<=10)
    
            alat = alat[cut_out]
            alon = alon[cut_out]
            ahh = ahh[cut_out]
            # aaz = aaz[cut_out]
            at = at[cut_out]
            ROTI = ROTI[cut_out]
            ROTI15 = ROTI15[cut_out]
            elev = elev[cut_out]
            #
            # # Resultados por saída padrão...
            # for i in range(ROTI.size):
            #     print("%13.8f %13.8f %13.8f %13.8f %16.10f %6.3f %s %s" % (alon[i], alat[i], ahh[i], at[i], elev[i], ROTI[i], estacao, sat1))  # ,OBS))
    
            # Inicialize um dicionário para armazenar os dados deste satélite
            dados_satelite = {
                'MJD': at,
                'Longitude': alon,
                'Latitude': alat,
                'Height': ahh,
                'Elevation': elev,
                'ROTI': ROTI,
                'ROTI15': ROTI15,
                'STA': estacao,
                'SAT': sat1
            }
    
            # Adicione o dicionário de dados do satélite à lista
            dados_satelites.append(dados_satelite)
    
        # Concatene todos os dicionários de dados em um DataFrame
        df_concatenado = pd.concat([pd.DataFrame(dados) for dados in dados_satelites], ignore_index=True)
    
        # Exiba o DataFrame concatenado
        #print(df_concatenado)    
    
        output_directory = os.path.join(destination_directory, estacao.upper())
        full_path = output_directory
        file_name = f"{estacao}_{doy}_{ano}_{satx}_ROTI.txt"
        output_file_path = os.path.join(full_path, file_name)
    
        # Garantir que o diretório exista
        os.makedirs(full_path, exist_ok=True)
    
        # Salvar o DataFrame selecionado em um arquivo de texto separado por tabulação
        df_concatenado.to_csv(output_file_path, sep='\t', index=False, na_rep='-999999.999')
    
    
        # Plotando o ROTI para o satélite atual com uma cor da paleta
        color = palette(idx / len(satellites_to_plot))
        plt.scatter(df_concatenado['MJD'], df_concatenado['ROTI'], marker='o', color='red')
        plt.scatter(df_concatenado['MJD'], df_concatenado['ROTI15'], marker='o', color='red',)
    
        # # Convertendo a lista de ROTI concatenado para um array numpy
        # ROTI_concatenado = np.array(ROTI_concatenado)
    
        # Supondo que 'mjd' seja uma lista de valores MJD em formato de string
        # Converta os valores MJD para objetos de data e hora
        start_time_mjd = min(map(float, mjd))
        start_time_datetime = datetime(1858, 11, 17) + timedelta(days=start_time_mjd)
        datetimes = [start_time_datetime + timedelta(days=float(at_val)) for at_val in mjd]
    
        # Configure o formato do eixo x para exibir apenas a hora e o minuto
        hours_fmt = mdates.DateFormatter('%H')
    
        # Configure o localizador de hora para o intervalo de 1 hora
        hour_locator = mdates.HourLocator(interval=2)
    
        # Configure o gráfico
        plt.gca().xaxis.set_major_formatter(hours_fmt)
        plt.gca().xaxis.set_major_locator(hour_locator)
    
        # Aumentando o tamanho dos rótulos do eixo x
        plt.xticks(fontsize=14)
    
    
        # Definindo título e rótulo do eixo y
        plt.ylabel('ROTI (TECU/min)', fontsize=16)
    
        # Definindo título e rótulo do eixo x
        plt.xlabel('Hora (UT)', fontsize=16)
    
        # Aumentando o tamanho dos rótulos do eixo x
        plt.xticks(fontsize=14)
    
        # Aumentando o tamanho dos rótulos do eixo x
        plt.yticks(fontsize=14)
    
        # Definindo título do gráfico
        plt.title(estacao, fontsize=18)
    
        plt.ylim(0,5)
        plt.grid(True)
        plt.tight_layout()
    
    
        file_name_png = f"{estacao}_{doy}_{ano}_ROTI.png"
        output_file_path_png = os.path.join(full_path, file_name_png)
    
        # Salvando a figura com 300 DPI
        plt.savefig(output_file_path_png, dpi=300)
    
    
    
        # Exiba o gráfico
        #plt.show()
    
        plt.gca().xaxis.set_major_formatter(hours_fmt)




