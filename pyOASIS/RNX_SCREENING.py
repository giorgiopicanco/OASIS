#!/usr/bin/env python3
from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import itertools
import warnings
from numpy.polynomial import Polynomial
from pyOASIS import gnss_freqs
from pyOASIS import screening_settings
from pyOASIS import settings
import pyOASIS

# Definindo o caminho do diretório principal
#diretorio_principal = "/media/debian-giorgio/DATA/GNSS_DATA/RINEX/PROCESSED"

def RNXScreening(destination_directory):
    
    # Lista de arquivos dentro do diretório .RNX1
    filess = os.listdir(destination_directory)

    # Filtrar apenas os arquivos .MAP
    files = [file_ for file_ in filess if file_.endswith("RNX1")]
    
    for file in files:

        # f = sys.argv[1]
        # destination_directory = sys.argv[2]
        f = os.path.join(destination_directory, file)
    
        # Redirecionar stderr para /dev/null
        sys.stderr = open(os.devnull, 'w')    
        
        # Obtendo o nome do arquivo
        g = os.path.basename(f)
        
        ano = g[13:17]
        doy = g[9:12]
        estacao = g[0:4]
        sat = g[5:8]
        # print(ano,doy,estacao,sat)
        #
        # sys.exit()
        
        # Variáveis e Parâmetros
        h1 = 0
        n_horas = 24 #horas
        int1 = 120 #minutos
        # f1 = 1575.42 * 10**6
        # f2 = 1227.60 * 10**6
        
        # Accessing the frequencies of the GPS system
        gps_freqs = gnss_freqs.FREQUENCY[gnss_freqs.GPS]
        f1 = gps_freqs[1]
        f2 = gps_freqs[2]
        f5 = gps_freqs[5]
        
        
        # Definir o nome do arquivo
        file_name = pyOASIS.__path__[0]+'/glonass_channels.dat'
        
        # Ler o arquivo para um DataFrame com os nomes das colunas definidos
        df_slots = pd.read_csv(file_name, sep=' ', header=None, names=['Slot', 'Channel'])
        glonass_frequencies = gnss_freqs.FREQUENCY[gnss_freqs.GLO]
        
        # Lista para armazenar os dados
        data = []
        
        # Iterar sobre cada linha do DataFrame
        for index, row in df_slots.iterrows():
            satellite = row['Slot']
            k = row['Channel']
            row_data = [satellite]
        
            for channel, frequency in glonass_frequencies.items():
                if callable(frequency):  # Verifica se é uma função lambda
                    freq_value = frequency(k)
                else:
                    freq_value = frequency
                formatted_freq = f"{freq_value:.1f}"
                row_data.append(formatted_freq)
        
            # Adicionar os dados da linha à lista de dados
            data.append(row_data)
        
        # Converter a lista de listas em um DataFrame pandas
        glo_freqs = pd.DataFrame(data, columns=['Satellite', 'fr1', 'fr2', 'fr3'])
        
        # Verificando a classe do satélite e ajustando os valores de f1, f2, f5
        if sat.startswith('G'):
            f1 = f1
            f2 = f2
            f5 = f5
        elif sat.startswith('R'):
            # Localizar a linha onde 'Satellite' é igual a 'sat'
            sat_row = glo_freqs.loc[glo_freqs['Satellite'] == sat]
        
            if not sat_row.empty:
                f1 = float(sat_row['fr1'].values[0])
                f2 = float(sat_row['fr2'].values[0])
                f5 = float(sat_row['fr3'].values[0])
        else:
            f1 = f2 = f5 = None  # Ou valores padrão
        
        
        # Inicializando listas para armazenar os valores de cada variável de todos os arquivos
        date = []
        time = []
        mjd = []
        pos_x = []
        pos_y = []
        pos_z = []
        L1 = []
        L2 = []
        L5 = []
        P1 = []
        P2 = []
        P5 = []
        cs_flags = []
        satellites = []
        sta = []
        hght = []
        El = []
        Lon = []
        Lat = []
        obs_La = []
        obs_Lb = []
        obs_Lc = []
        obs_Ca = []
        obs_Cb = []
        obs_Cc = []
        
        # for arquivo in arquivos:
        caminho_arquivo = f
        
        print("aqui", caminho_arquivo)
        # print("Conteúdo do arquivo", arquivo)
        with open(caminho_arquivo, 'r') as f:
            # Lendo o cabeçalho do arquivo
            header = f.readline().strip().split('\t')
            obs_La_header = header[6]
            obs_Lb_header = header[7]
            obs_Lc_header = header[8]
            obs_Ca_header = header[9]
            obs_Cb_header = header[10]
            obs_Cc_header = header[11]
            #
            # print(obs_La_header,obs_Lb_header, obs_Lc_header, obs_Ca_header, obs_Cb_header, obs_Cc_header)
        
        
        
            # Lendo cada linha de dados do arquivo
            for linha in f:
                # print(linha)
                # Dividindo a linha em colunas
                colunas = linha.strip().split('\t')  # Supondo que as colunas estejam separadas por tabulação (\t)
                # Associando cada coluna com o cabeçalho correspondente
                registro = {
                    'date': colunas[0],
                    'time': colunas[1],
                    'mjd': colunas[2],
                    'pos_x': colunas[3],
                    'pos_y': colunas[4],
                    'pos_z': colunas[5],
                    'L1': colunas[6],
                    'L2': colunas[7],
                    'L5': colunas[8],
                    'P1': colunas[9],
                    'P2': colunas[10],
                    'P5': colunas[11],
                    'cs_flags': colunas[12],
                    'satellite': colunas[13],
                    'sta': colunas[14],
                    'hght': colunas[15],
                    'El': colunas[16],
                    'Lon': colunas[17],
                    'Lat': colunas[18],
                    'obs_La': obs_La_header,
                    'obs_Lb': obs_Lb_header,
                    'obs_Lc': obs_Lc_header,
                    'obs_Ca': obs_Ca_header,
                    'obs_Cb': obs_Cb_header,
                    'obs_Cc': obs_Cc_header
                }
                # Adicionando os valores de cada variável às respectivas listas
                # timestamp.append(registro['timestamp'])
        
                date.append(registro['date'])
                time.append(registro['time'])
                mjd.append(registro['mjd'])
                pos_x.append(registro['pos_x'])
                pos_y.append(registro['pos_y'])
                pos_z.append(registro['pos_z'])
                L1.append(registro['L1'])
                L2.append(registro['L2'])
                L5.append(registro['L5'])
                P1.append(registro['P1'])
                P2.append(registro['P2'])
                P5.append(registro['P5'])
                cs_flags.append(registro['cs_flags'])
                satellites.append(registro['satellite'])
                sta.append(registro['sta'])
                hght.append(registro['hght'])
                El.append(registro['El'])
                Lon.append(registro['Lon'])
                Lat.append(registro['Lat'])
                obs_La.append(registro['obs_La'])
                obs_Lb.append(registro['obs_Lb'])
                obs_Lc.append(registro['obs_Lc'])
                obs_Ca.append(registro['obs_Ca'])
                obs_Cb.append(registro['obs_Cb'])
                obs_Cc.append(registro['obs_Cc'])
        
        
        # sat = 'G'
        # # Filtrando os satélites
        # if sat:
        #     satellites_to_plot = [sv for sv in np.unique(satellites) if sv.startswith(sat)]
        # else:
        #     satellites_to_plot = np.unique(satellites)
        
        
        # Inicializando listas para armazenar os valores de cada variável de todos os arquivos
        date_filtered = []
        time_filtered = []
        mjd_filtered = []
        pos_x_filtered = []
        pos_y_filtered = []
        pos_z_filtered = []
        L1_filtered = []
        L2_filtered = []
        L5_filtered = []
        P1_filtered = []
        P2_filtered = []
        P5_filtered = []
        cs_flags_filtered = []
        satellites_filtered = []
        sta_filtered = []
        hght_filtered = []
        El_filtered = []
        Lon_filtered = []
        Lat_filtered = []
        obs_La_filtered = []
        obs_Lb_filtered = []
        obs_Lc_filtered = []
        obs_Ca_filtered = []
        obs_Cb_filtered = []
        obs_Cc_filtered = []
        
        satellite = sat
        print(satellite)
        indices = np.where(np.array(satellites) == satellite)[0]
        
        date_filtered = []
        time_filtered = []
        mjd_filtered = []
        pos_x_filtered = []
        pos_y_filtered = []
        pos_z_filtered = []
        L1_filtered = []
        L2_filtered = []
        L5_filtered = []
        P1_filtered = []
        P2_filtered = []
        P5_filtered = []
        cs_flags_filtered = []
        satellites_filtered = []
        sta_filtered = []
        hght_filtered = []
        El_filtered = []
        Lon_filtered = []
        Lat_filtered = []
        obs_La_filtered = []
        obs_Lb_filtered = []
        obs_Lc_filtered = []
        obs_Ca_filtered = []
        obs_Cb_filtered = []
        obs_Cc_filtered = []
        
        
        for idx in indices:
            date_filtered.append(date[idx])
            time_filtered.append(time[idx])
            mjd_filtered.append(mjd[idx])
            pos_x_filtered.append(pos_x[idx])
            pos_y_filtered.append(pos_y[idx])
            pos_z_filtered.append(pos_z[idx])
            L1_filtered.append(L1[idx])
            L2_filtered.append(L2[idx])
            L5_filtered.append(L5[idx])
            P1_filtered.append(P1[idx])
            P2_filtered.append(P2[idx])
            P5_filtered.append(P5[idx])
            cs_flags_filtered.append(cs_flags[idx])
            satellites_filtered.append(satellites[idx])
            sta_filtered.append(sta[idx])
            hght_filtered.append(hght[idx])
            El_filtered.append(El[idx])
            Lon_filtered.append(Lon[idx])
            Lat_filtered.append(Lat[idx])
            obs_La_filtered.append(obs_La[idx])
            obs_Lb_filtered.append(obs_Lb[idx])
            obs_Lc_filtered.append(obs_Lc[idx])
            obs_Ca_filtered.append(obs_Ca[idx])
            obs_Cb_filtered.append(obs_Cb[idx])
            obs_Cc_filtered.append(obs_Cc[idx])
        
        
        
        # Construindo um DataFrame com os dados filtrados para cada satélite
        data = {
            'date': date_filtered,
            'time2': time_filtered,
            'mjd': mjd_filtered,
            'pos_x': pos_x_filtered,
            'pos_y': pos_y_filtered,
            'pos_z': pos_z_filtered,
            'L1': L1_filtered,
            'L2': L2_filtered,
            'L5': L5_filtered,
            'P1': P1_filtered,
            'P2': P2_filtered,
            'P5': P2_filtered,
            'cs_flag': cs_flags_filtered,
            'satellite': satellites_filtered,
            'sta': sta_filtered,
            'hght': hght_filtered,
            'El': El_filtered,
            'Lon': Lon_filtered,
            'Lat': Lat_filtered,
            'obs_La': obs_La_filtered,
            'obs_Lb': obs_Lb_filtered,
            'obs_Lc': obs_Lc_filtered,
            'obs_Ca': obs_Ca_filtered,
            'obs_Cb': obs_Cb_filtered,
            'obs_Cc': obs_Cc_filtered
        }
        
        df = pd.DataFrame(data)
        # print(df)
        # Convertendo as colunas relevantes para float
        columns_to_convert = ['L1', 'L2', 'L5', 'P1', 'P2', 'P5']
        df[columns_to_convert] = df[columns_to_convert].astype(float)
        
        # Substituir -999999.999 por NaN nas colunas relevantes
        df.replace(-999999.999, np.nan, inplace=True)
        
        
        # Convertendo as colunas 'date' e 'time' para o tipo datetime e depois concatenando
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time2'])
        
        # Convertendo a coluna 'timestamp' para o tipo datetime, se ainda não estiver
        # df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # # Extraindo apenas o tempo de 'timestamp' e armazenando em uma nova coluna 'time'
        df['time'] = df['timestamp'].dt.time
        
        # Convertendo as listas em arrays numpy e garantindo que os tipos de dados sejam float64
        L1_array = np.nan_to_num(np.array(df['L1'].tolist(), dtype=np.float64), nan=-999999.999)
        L2_array = np.nan_to_num(np.array(df['L2'].tolist(), dtype=np.float64), nan=-999999.999)
        L5_array = np.nan_to_num(np.array(df['L5'].tolist(), dtype=np.float64), nan=-999999.999)
        
        P1_array = np.nan_to_num(np.array(df['P1'].tolist(), dtype=np.float64), nan=-999999.999)
        P2_array = np.nan_to_num(np.array(df['P2'].tolist(), dtype=np.float64), nan=-999999.999)
        P5_array = np.nan_to_num(np.array(df['P5'].tolist(), dtype=np.float64), nan=-999999.999)
        
        # Substituindo -999999.999 por "NaN" em P1_array
        L1_array[L1_array == -999999.999] = np.nan
        L2_array[L2_array == -999999.999] = np.nan
        L5_array[L5_array == -999999.999] = np.nan
        
        P1_array[P1_array == -999999.999] = np.nan
        P2_array[P2_array == -999999.999] = np.nan
        P5_array[P5_array == -999999.999] = np.nan
        
        # Calculando a combinação de Melbourne-Wubbena para o satélite atual
        MW_combination = screening_settings.melbourne_wubbena_combination(f1, f2, L1_array, L2_array, P1_array, P2_array)
        MW_combination2 = screening_settings.melbourne_wubbena_combination(f1, f5, L1_array, L5_array, P1_array, P5_array)
        
        
        IFL_combination = screening_settings.iono_free_phase_combination(f1, f2, L1_array, L2_array)
        IFP_combination = screening_settings.iono_free_range_combination(f1, f2, P1_array, P2_array)
        
        # # for value in IFL_combination:
        # #     print(value)
        
        
        # Adicionando a combinação de Melbourne-Wubbena, Ionosphere-Free (phase) e Ionosphere-Free (code) ao DataFrame
        df['MW'] = MW_combination
        df['MW2'] = MW_combination2
        
        
        # Suponho que df['cs_flag'] seja uma série do pandas
        arcos = []  # Lista para armazenar os arcos de observação
        arc_atual = []  # Lista temporária para armazenar o arco de observação atual
        
        # Iterar sobre todos os elementos de df['cs_flag']
        for idx, value in enumerate(df['cs_flag']):
            if value == 'S':
                # Se o valor atual for 'S', verificar se o arco atual está vazio
                # Isso evita adicionar arcos vazios caso haja 'S's consecutivos
                if arc_atual:
                    arcos.append(arc_atual)
                    arc_atual = []  # Reseta a lista do arco atual
            else:
                # Se o valor não for 'S', adicionar o índice ao arco atual
                arc_atual.append(idx)
        
        # Adicionar o último arco se ele não estiver vazio
        if arc_atual:
            arcos.append(arc_atual)
        
        print()
        
        # Imprimir informações de cada arco e classificá-los
        for i, arc in enumerate(arcos):
            start_index = arc[0]
            end_index = arc[-1]
            num_observations = len(arc)
            status = "Mantido" if num_observations >= 15 else "Descartado"
            print(f"Arco {i + 1}: {df['timestamp'][start_index]} - {df['timestamp'][end_index]}, Start = {start_index}, End = {end_index}, "
                f"Obs. = {num_observations}, Status = {status}")
        
        # print(len(arcos))
        # sys.exit()
        # Separar os dados MW_combination em cada arco
        arc_data = []
        arc_idx = []
        polynomial_fits = []
        
        print()
        
        for i, arc in enumerate(arcos):
            start = arc[0]
            end = arc[-1]
        
            arc_values = MW_combination[start:end+1]
            arc_timestamps = df['timestamp'][start:end+1]
        
            if len(arc_values) < 15:
                continue
        
            # Ajustar um polinômio de segundo grau
            x_values = np.arange(len(arc_values))
            polynomial_fit = screening_settings.fit_polynomial(x_values, arc_values, 3)
        
            # Armazenar os dados do arco e o ajuste do polinômio
            arc_data.append(arc_values)
            arc_idx.append(arc_timestamps)
            polynomial_fits.append(polynomial_fit)
        
            # Imprimir informações do arco
            num_observations = len(arc_values)
            num_points_fit = len(polynomial_fit)
        
            print(f"Arco {i + 1}: Índice inicial = {start}, Índice final = {end}, "
                f"Número de observações = {num_observations}, Número de pontos do ajuste = {num_points_fit}")
        
        #
        # sys.exit()
        
        # Filtrar os arcos que passaram no critério de comprimento
        arcos_validos = [arc for arc in arcos if len(MW_combination[arc[0]:arc[-1]+1]) >= 15]
        
        # Se houver apenas um arco válido, duplicar para que haja dois arcos válidos
        if len(arcos_validos) == 1:
            arcos_validos.append(arcos_validos[0])
        
        # Definir o número máximo de colunas por linha
        max_colunas_por_linha = 2
        
        # Calcular o número de linhas e colunas necessárias
        num_arcos_validos = len(arcos_validos)
        
        num_linhas = (num_arcos_validos - 1) // max_colunas_por_linha + 1
        num_colunas = min(num_arcos_validos, max_colunas_por_linha)
        
        # Criar a grade de subplots
        fig, axes = plt.subplots(num_linhas, num_colunas, figsize=(6*num_colunas, 5*num_linhas))
        
        all_all = []
        
        all_index = []
        
        # Iterar sobre cada arco válido e plotar os dados
        for i, (arc, ax) in enumerate(zip(arcos_validos, axes.flatten()), start=1):
        
            start = arc[0]
            end = arc[-1]
        
            arc_data = df.iloc[arc]
            time = df.index[arc]
        
            arc_values = MW_combination[start:end+1]
            # arc_values = IFL_combination[start:end+1]
            arc_timestamps = df['timestamp'][start:end+1]
            arc_values2 = arc_values
        
            # Calcular o tempo decorrido em segundos desde o primeiro timestamp do arco
            x = (arc_timestamps - arc_timestamps.iloc[0]).dt.total_seconds()
        
            y_rescaled = screening_settings.rescale_data(arc_values)
            delta_y = np.diff(y_rescaled, prepend=np.nan)
        
            # Ajustar um polinômio apenas nos valores válidos (excluindo np.nan)
            p = Polynomial.fit(x[1:], delta_y[1:], 3)
        
            delta_y_fit = p(x)  # Valores ajustados pelo polinômio
            # residuals = abs(delta_y - delta_y_fit)  # Calcular resíduos
            residuals = delta_y - delta_y_fit  # Calcular resíduos
        
            mini_arcos = []  # Lista para armazenar os arcos de observação
            mini_arcos_mantidos = []
            mini_arc_atual = []  # Lista temporária para armazenar o arco de observação atual
            signo_anterior = None
        
            # Iterar sobre todos os elementos de residuals
            for idx, value in enumerate(residuals):
                if signo_anterior is None:  # Se for o primeiro valor, inicializa o signo_anterior
                    signo_anterior = np.sign(value)
        
                # Verifica se o sinal mudou
                if np.sign(value) != signo_anterior:
                    # Se o arco atual não estiver vazio, adiciona-o à lista de mini-arcos
                    if mini_arc_atual:
                        mini_arcos.append(mini_arc_atual)
                    mini_arc_atual = []  # Inicia um novo mini-arco
        
                # Adiciona o índice ao mini-arco atual
                mini_arc_atual.append(idx)
        
                # Atualiza o sinal anterior
                signo_anterior = np.sign(value)
        
            # Lista para armazenar os miniarcos que passam no critério positivo
            mini_arcos_mantidos = []
        
            # Definindo uma flag para controlar a saída do loop
            should_break = False
        
            print()
            print("Looking for mini cycle-slips in L1-L2 pair:")
            print()
        
            # Loop externo
            for mini_i, mini_arc in enumerate(mini_arcos):
                mini_start_index = mini_arc[0]
                mini_end_index = mini_arc[-1]
                num_observations = len(mini_arc)
                status = "Mantido" if num_observations >= 4 else "Descartado"
                print(f"Mini-arco {mini_i + 1}: Start = {mini_start_index}, End = {mini_end_index}, Obs. = {num_observations}, Status = {status}")
        
                # Verifica se o número de observações é menor que 4
                if num_observations <= 4:
                    # Atualiza a flag para indicar que devemos sair dos loops
                    should_break = True
                    continue
                    # break  # Sai do loop interno
        
                # Se o miniarco tem pelo menos 4 observações, adiciona-o à lista de miniarcos mantidos
                mini_arcos_mantidos.append(mini_arc)
            # sys.exit()
            # Se a flag indicar que devemos sair dos loops, saímos do loop externo também
            if should_break:
                continue
        #
            # Imprimir os miniarcos mantidos
            print("Miniarcos mantidos:")
            for i, mini_arc in enumerate(mini_arcos_mantidos):
                print(f"Mini-arco {i + 1}: {mini_arc}")
        
        
            print('TESTE OK')
        
        
        
        
            # Cálculo dos quartis e IQR para identificação de outliers
            Q1 = np.nanpercentile(residuals, 15)
            Q3 = np.nanpercentile(residuals, 85)
            IQR = Q3 - Q1
        
            threshold2 = 2
        
            outlier_mask = (residuals < Q1 - threshold2 * IQR) | (residuals > Q3 + threshold2 * IQR)
            high_residuals_mask = residuals > 1  # Máscara para resíduos altos
            other_residuals_mask = ~(outlier_mask | high_residuals_mask)  # Máscara para os demais resíduos
        
            # Ajustar um polinômio de segundo grau
            x_values = np.arange(len(arc_values))
            polynomial_fit = screening_settings.fit_polynomial(x_values, arc_values, 3)
        
            # Plotar os dados do arco
            ax.scatter(arc_timestamps, arc_values, label='Dados', color='blue', s=15)
        
            # Plotar o ajuste polinomial
            ax.plot(arc_timestamps, polynomial_fit, label='Ajuste', color='red')
        
        
            # Definir o formato do eixo x como horas
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        
            # Obtendo o número do arco válido correspondente
            num_arc_valido = arcos.index(arc) + 1
        
            # Adicionar o número do arco válido ao título
            ax.set_title(f"Arco {num_arc_valido}")
        
            # Adicionar subplots menores dentro dos subplots principais
            ax_fit = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
            ax_residuals = ax.inset_axes([0.6, 0.15, 0.35, 0.35])
        
            # # Cores para cada mini-arco
            cores = ['lightblue', 'yellow', 'lime', 'green', 'purple', 'navy', 'blue','lime', 'green', 'navy']
        
        
        
            all_out = []  # Inicializar a lista 'all_out' antes do loop
            all__all = []
            # Inicialize a lista para armazenar os índices de resíduos altos e outliers
            all_indices = []
        
            # Lista para armazenar todos os índices
            todos_indices = []
        
        
        
        
            for i, (mini_start, mini_end) in enumerate(mini_arcos_mantidos):
        
        
        
        
                mini_residuals = residuals[mini_start:mini_end]
                mini_time = time[mini_start:mini_end]
        
                # print(len(mini_residuals))
        
        
        
                try:
                    mini_fit = screening_settings.fit_polynomial(mini_time, mini_residuals, 3)
        
                    new_mini_residuals = abs(mini_residuals-mini_fit)
        
        
        
                    # Cálculo dos quartis e IQR para identificação de outliers
                    mini_Q1 = np.nanpercentile(new_mini_residuals, 15)
                    mini_Q3 = np.nanpercentile(new_mini_residuals, 85)
                    mini_IQR = mini_Q3 - mini_Q1
        
                    mini_threshold2 = 1.3
        
                    mini_outlier_mask = (new_mini_residuals < mini_Q1 - mini_threshold2 * mini_IQR) | (new_mini_residuals > mini_Q3 + mini_threshold2 * mini_IQR)
                    mini_high_residuals_mask = new_mini_residuals > 0.0002  # Máscara para resíduos altos
                    mini_other_residuals_mask = ~(mini_outlier_mask | mini_high_residuals_mask)  # Máscara para os demais resíduos
        
                    micro_residuals = abs(np.diff(new_mini_residuals, prepend=np.nan))
        
                    # Calcule os quartis
                    q1_micro = np.nanpercentile(micro_residuals, 15)
                    q3_micro = np.nanpercentile(micro_residuals, 85)
        
                    # Calcule a amplitude interquartil
                    iqr_micro = q3_micro - q1_micro
        
                    micro_threshold2 = 1.3
        
                    # Defina os limites para identificar outliers
                    lower_bound_micro = q1_micro - micro_threshold2 * iqr_micro
                    upper_bound_micro = q3_micro + micro_threshold2 * iqr_micro
        
                    # Encontre outliers
                    outliers_micro = (micro_residuals < lower_bound_micro) | (micro_residuals > upper_bound_micro)
        
        
                    micro_high_residuals_mask = micro_residuals > 0.00002  # Máscara para resíduos altos
        
        
                    ax_fit.scatter(mini_time, abs(micro_residuals), color=cores[i], s=2, label='Out.', marker='o')
                    ax_fit.scatter(mini_time[micro_high_residuals_mask], micro_residuals[micro_high_residuals_mask], color='red', s=2, label='Out.', marker='o')
                    ax_fit.scatter(mini_time[outliers_micro], abs(micro_residuals[outliers_micro]), color='red', s=2, label='Out.', marker='o')
        
        
        
        
                    # Convertendo os índices locais para índices globais - mini
                    global_high_residuals_indices = np.where(mini_high_residuals_mask)[0] + mini_start
                    global_outlier_indices = np.where(mini_outlier_mask)[0] + mini_start
        
                    # Convertendo os índices locais para índices globais - micro
                    global_high_residuals_indices_micro = np.where(micro_high_residuals_mask)[0] + mini_start
                    global_outlier_indices_micro = np.where(outliers_micro)[0] + mini_start
        
        
                    # ax_residuals.scatter(mini_time, new_mini_residuals, color=cores[i], s=1, label='Out.', marker='o')
                    ax_residuals.scatter(mini_time[mini_outlier_mask], new_mini_residuals[mini_outlier_mask], edgecolor='red', facecolor='none', s=2, label='Out.', marker='o')
                    ax_residuals.scatter(mini_time[outliers_micro], new_mini_residuals[outliers_micro], edgecolor='red', facecolor='none', s=2, label='Out.', marker='o')
                    ax_residuals.scatter(mini_time[micro_high_residuals_mask], new_mini_residuals[micro_high_residuals_mask], edgecolor='pink', facecolor='none', s=2, label='Out.', marker='o')
        
                    print('OKOKOKO')
        
                    # Listas para armazenar os índices de resíduos altos e outliers
                    indices_residuos_altos_mini = start + global_high_residuals_indices
                    indices_outliers_mini = start + global_outlier_indices
                    indices_residuos_altos_micro = start + global_high_residuals_indices_micro
                    indices_outliers_micro = start + global_outlier_indices_micro
        
        
                    # Lista para armazenar os índices de resíduos altos e outliers
                    indices_residuos_altos_mini = start + global_high_residuals_indices
                    indices_outliers_mini = start + global_outlier_indices
                    indices_residuos_altos_micro = start + global_high_residuals_indices_micro
                    indices_outliers_micro = start + global_outlier_indices_micro
        
                    # Adiciona os índices à lista geral
                    todos_indices.append({
                        'residuos_altos_mini': indices_residuos_altos_mini,
                        'outliers_mini': indices_outliers_mini,
                        'residuos_altos_micro': indices_residuos_altos_micro,
                        'outliers_micro': indices_outliers_micro
                    })
        
        
        
                except:
                    pass  # Ignorar e continuar o loop
                    # print('ERRO')
        
        
        
        
        
            # Remover subplots não utilizados
            for i in range(num_arcos_validos, num_linhas*num_colunas):
                fig.delaxes(axes.flatten()[i])
        
        
            # plt.show()
        
        
        
        
        
            # Criar uma cópia da lista todos_indices antes de estender todos_indices_vertical
            todos_indices_copy = todos_indices.copy()
        
            # # Agora você pode acessar todos os índices fora do loop
            for indices in todos_indices_copy:
                print(f"Índices para iteração {i}:")
                print("Resíduos Altos para Mini-Arco:", indices['residuos_altos_mini'])
                print("Outliers para Mini-Arco:", indices['outliers_mini'])
                print("Resíduos Altos para Micro-Arco:", indices['residuos_altos_micro'])
                print("Outliers para Micro-Arco:", indices['outliers_micro'])
                print()
        
        
        
            # Concatena todos os índices em uma lista vertical
            todos_indices_vertical = []
            for indices in todos_indices_copy:
                todos_indices_vertical.extend(indices['residuos_altos_mini'])
                todos_indices_vertical.extend(indices['outliers_mini'])
                todos_indices_vertical.extend(indices['residuos_altos_micro'])
                todos_indices_vertical.extend(indices['outliers_micro'])
        
        
        
            from collections import OrderedDict
        
            # Converta todos_indices_vertical em um conjunto para remover duplicatas e, em seguida, converta de volta para uma lista mantendo a ordem original
            todos_indices_vertical = list(OrderedDict.fromkeys(todos_indices_vertical))
        
        
                # Remover subplots não utilizados
            for i in range(num_arcos_validos, num_linhas*num_colunas):
                fig.delaxes(axes.flatten()[i])
        
        
            # print(todos_indices_vertical)
        
            all_index.extend(todos_indices_vertical)
        
            # Remover espaços vazios da lista
            all_index = list(filter(None, all_index))
        
        
        
        
        
        
        
        # # Agora você pode acessar todos os índices em uma lista vertical
        # print('Todos os índices em uma lista vertical:')
        #
        # for value in all_index:
        #     print(value)
        print()
        print(satellite)
        
        df['outlier_flag'] = 'N'
        
        # Substituir 'N' por 'Y' nos índices especificados
        df.loc[all_index, 'outlier_flag'] = 'Y'
        
        
        
        # for i, value in enumerate(df['outlier_flag']):
        #     print(i,value)
        
        
        
        
        # ----------------- [L1 - L5]
        
        all_index15 = []
        
        # Iterar sobre cada arco válido e plotar os dados
        for i, (arc, ax) in enumerate(zip(arcos_validos, axes.flatten()), start=1):
        
            start = arc[0]
            end = arc[-1]
        
            arc_data = df.iloc[arc]
            time = df.index[arc]
        
            arc_values = MW_combination2[start:end+1]
            # arc_values = IFL_combination[start:end+1]
            arc_timestamps = df['timestamp'][start:end+1]
            arc_values2 = arc_values
        
            # Calcular o tempo decorrido em segundos desde o primeiro timestamp do arco
            x = (arc_timestamps - arc_timestamps.iloc[0]).dt.total_seconds()
        
            y_rescaled = screening_settings.rescale_data(arc_values)
            delta_y = np.diff(y_rescaled, prepend=np.nan)
        
            # Ajustar um polinômio apenas nos valores válidos (excluindo np.nan)
            p = Polynomial.fit(x[1:], delta_y[1:], 3)
        
            delta_y_fit = p(x)  # Valores ajustados pelo polinômio
            # residuals = abs(delta_y - delta_y_fit)  # Calcular resíduos
            residuals = delta_y - delta_y_fit  # Calcular resíduos
        
            mini_arcos = []  # Lista para armazenar os arcos de observação
            mini_arcos_mantidos = []
            mini_arc_atual = []  # Lista temporária para armazenar o arco de observação atual
            signo_anterior = None
        
            # Iterar sobre todos os elementos de residuals
            for idx, value in enumerate(residuals):
                if signo_anterior is None:  # Se for o primeiro valor, inicializa o signo_anterior
                    signo_anterior = np.sign(value)
        
                # Verifica se o sinal mudou
                if np.sign(value) != signo_anterior:
                    # Se o arco atual não estiver vazio, adiciona-o à lista de mini-arcos
                    if mini_arc_atual:
                        mini_arcos.append(mini_arc_atual)
                    mini_arc_atual = []  # Inicia um novo mini-arco
        
                # Adiciona o índice ao mini-arco atual
                mini_arc_atual.append(idx)
        
                # Atualiza o sinal anterior
                signo_anterior = np.sign(value)
        
            # Lista para armazenar os miniarcos que passam no critério positivo
            mini_arcos_mantidos = []
        
            # Definindo uma flag para controlar a saída do loop
            should_break = False
        
            print()
            print("Looking for mini cycle-slips in L1-L5 pair:")
            print()
        
            # Loop externo
            for mini_i, mini_arc in enumerate(mini_arcos):
                mini_start_index = mini_arc[0]
                mini_end_index = mini_arc[-1]
                num_observations = len(mini_arc)
                status = "Mantido" if num_observations >= 4 else "Descartado"
                print(f"Mini-arco {mini_i + 1}: Start = {mini_start_index}, End = {mini_end_index}, Obs. = {num_observations}, Status = {status}")
        
                # Verifica se o número de observações é menor que 4
                if num_observations <= 4:
                    # Atualiza a flag para indicar que devemos sair dos loops
                    should_break = True
                    continue
                    # break  # Sai do loop interno
        
                # Se o miniarco tem pelo menos 4 observações, adiciona-o à lista de miniarcos mantidos
                mini_arcos_mantidos.append(mini_arc)
            # sys.exit()
            # Se a flag indicar que devemos sair dos loops, saímos do loop externo também
            if should_break:
                continue
        #
            # Imprimir os miniarcos mantidos
            print("Miniarcos mantidos:")
            for i, mini_arc in enumerate(mini_arcos_mantidos):
                print(f"Mini-arco {i + 1}: {mini_arc}")
        
        
            print('TESTE OK')
        
        
        
        
            # Cálculo dos quartis e IQR para identificação de outliers
            Q1 = np.nanpercentile(residuals, 15)
            Q3 = np.nanpercentile(residuals, 85)
            IQR = Q3 - Q1
        
            threshold2 = 2
        
            outlier_mask = (residuals < Q1 - threshold2 * IQR) | (residuals > Q3 + threshold2 * IQR)
            high_residuals_mask = residuals > 1  # Máscara para resíduos altos
            other_residuals_mask = ~(outlier_mask | high_residuals_mask)  # Máscara para os demais resíduos
        
            # Ajustar um polinômio de segundo grau
            x_values = np.arange(len(arc_values))
            polynomial_fit = screening_settings.fit_polynomial(x_values, arc_values, 3)
        
            # Plotar os dados do arco
            ax.scatter(arc_timestamps, arc_values, label='Dados', color='blue', s=15)
        
            # Plotar o ajuste polinomial
            ax.plot(arc_timestamps, polynomial_fit, label='Ajuste', color='red')
        
        
            # Definir o formato do eixo x como horas
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        
            # Obtendo o número do arco válido correspondente
            num_arc_valido = arcos.index(arc) + 1
        
            # Adicionar o número do arco válido ao título
            ax.set_title(f"Arco {num_arc_valido}")
        
            # Adicionar subplots menores dentro dos subplots principais
            ax_fit = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
            ax_residuals = ax.inset_axes([0.6, 0.15, 0.35, 0.35])
        
            # # Cores para cada mini-arco
            cores = ['lightblue', 'yellow', 'lime', 'green', 'purple', 'navy', 'blue','lime', 'green', 'navy']
        
        
        
            all_out = []  # Inicializar a lista 'all_out' antes do loop
            all__all = []
            # Inicialize a lista para armazenar os índices de resíduos altos e outliers
            all_indices = []
        
            # Lista para armazenar todos os índices
            todos_indices = []
        
        
        
        
            for i, (mini_start, mini_end) in enumerate(mini_arcos_mantidos):
        
        
        
        
                mini_residuals = residuals[mini_start:mini_end]
                mini_time = time[mini_start:mini_end]
        
                # print(len(mini_residuals))
        
        
        
                try:
                    mini_fit = screening_settings.fit_polynomial(mini_time, mini_residuals, 3)
        
                    new_mini_residuals = abs(mini_residuals-mini_fit)
        
        
        
                    # Cálculo dos quartis e IQR para identificação de outliers
                    mini_Q1 = np.nanpercentile(new_mini_residuals, 15)
                    mini_Q3 = np.nanpercentile(new_mini_residuals, 85)
                    mini_IQR = mini_Q3 - mini_Q1
        
                    mini_threshold2 = 1.3
        
                    mini_outlier_mask = (new_mini_residuals < mini_Q1 - mini_threshold2 * mini_IQR) | (new_mini_residuals > mini_Q3 + mini_threshold2 * mini_IQR)
                    mini_high_residuals_mask = new_mini_residuals > 0.0002  # Máscara para resíduos altos
                    mini_other_residuals_mask = ~(mini_outlier_mask | mini_high_residuals_mask)  # Máscara para os demais resíduos
        
                    micro_residuals = abs(np.diff(new_mini_residuals, prepend=np.nan))
        
                    # Calcule os quartis
                    q1_micro = np.nanpercentile(micro_residuals, 15)
                    q3_micro = np.nanpercentile(micro_residuals, 85)
        
                    # Calcule a amplitude interquartil
                    iqr_micro = q3_micro - q1_micro
        
                    micro_threshold2 = 1.3
        
                    # Defina os limites para identificar outliers
                    lower_bound_micro = q1_micro - micro_threshold2 * iqr_micro
                    upper_bound_micro = q3_micro + micro_threshold2 * iqr_micro
        
                    # Encontre outliers
                    outliers_micro = (micro_residuals < lower_bound_micro) | (micro_residuals > upper_bound_micro)
        
        
                    micro_high_residuals_mask = micro_residuals > 0.00002  # Máscara para resíduos altos
        
        
                    ax_fit.scatter(mini_time, abs(micro_residuals), color=cores[i], s=2, label='Out.', marker='o')
                    ax_fit.scatter(mini_time[micro_high_residuals_mask], micro_residuals[micro_high_residuals_mask], color='red', s=2, label='Out.', marker='o')
                    ax_fit.scatter(mini_time[outliers_micro], abs(micro_residuals[outliers_micro]), color='red', s=2, label='Out.', marker='o')
        
        
        
        
                    # Convertendo os índices locais para índices globais - mini
                    global_high_residuals_indices = np.where(mini_high_residuals_mask)[0] + mini_start
                    global_outlier_indices = np.where(mini_outlier_mask)[0] + mini_start
        
                    # Convertendo os índices locais para índices globais - micro
                    global_high_residuals_indices_micro = np.where(micro_high_residuals_mask)[0] + mini_start
                    global_outlier_indices_micro = np.where(outliers_micro)[0] + mini_start
        
        
                    # ax_residuals.scatter(mini_time, new_mini_residuals, color=cores[i], s=1, label='Out.', marker='o')
                    ax_residuals.scatter(mini_time[mini_outlier_mask], new_mini_residuals[mini_outlier_mask], edgecolor='red', facecolor='none', s=2, label='Out.', marker='o')
                    ax_residuals.scatter(mini_time[outliers_micro], new_mini_residuals[outliers_micro], edgecolor='red', facecolor='none', s=2, label='Out.', marker='o')
                    ax_residuals.scatter(mini_time[micro_high_residuals_mask], new_mini_residuals[micro_high_residuals_mask], edgecolor='pink', facecolor='none', s=2, label='Out.', marker='o')
        
                    print('OKOKOKO')
        
                    # Listas para armazenar os índices de resíduos altos e outliers
                    indices_residuos_altos_mini = start + global_high_residuals_indices
                    indices_outliers_mini = start + global_outlier_indices
                    indices_residuos_altos_micro = start + global_high_residuals_indices_micro
                    indices_outliers_micro = start + global_outlier_indices_micro
        
        
                    # Lista para armazenar os índices de resíduos altos e outliers
                    indices_residuos_altos_mini = start + global_high_residuals_indices
                    indices_outliers_mini = start + global_outlier_indices
                    indices_residuos_altos_micro = start + global_high_residuals_indices_micro
                    indices_outliers_micro = start + global_outlier_indices_micro
        
                    # Adiciona os índices à lista geral
                    todos_indices.append({
                        'residuos_altos_mini': indices_residuos_altos_mini,
                        'outliers_mini': indices_outliers_mini,
                        'residuos_altos_micro': indices_residuos_altos_micro,
                        'outliers_micro': indices_outliers_micro
                    })
        
        
        
                except:
                    pass  # Ignorar e continuar o loop
                    # print('ERRO')
        
        
        
        
        
            # Remover subplots não utilizados
            for i in range(num_arcos_validos, num_linhas*num_colunas):
                fig.delaxes(axes.flatten()[i])
        
        
            # plt.show()
        
        
        
        
        
            # Criar uma cópia da lista todos_indices antes de estender todos_indices_vertical
            todos_indices_copy = todos_indices.copy()
        
            # # Agora você pode acessar todos os índices fora do loop
            for indices in todos_indices_copy:
                print(f"Índices para iteração {i}:")
                print("Resíduos Altos para Mini-Arco:", indices['residuos_altos_mini'])
                print("Outliers para Mini-Arco:", indices['outliers_mini'])
                print("Resíduos Altos para Micro-Arco:", indices['residuos_altos_micro'])
                print("Outliers para Micro-Arco:", indices['outliers_micro'])
                print()
        
        
        
            # Concatena todos os índices em uma lista vertical
            todos_indices_vertical = []
            for indices in todos_indices_copy:
                todos_indices_vertical.extend(indices['residuos_altos_mini'])
                todos_indices_vertical.extend(indices['outliers_mini'])
                todos_indices_vertical.extend(indices['residuos_altos_micro'])
                todos_indices_vertical.extend(indices['outliers_micro'])
        
        
        
            from collections import OrderedDict
        
            # Converta todos_indices_vertical em um conjunto para remover duplicatas e, em seguida, converta de volta para uma lista mantendo a ordem original
            todos_indices_vertical = list(OrderedDict.fromkeys(todos_indices_vertical))
        
        
                # Remover subplots não utilizados
            for i in range(num_arcos_validos, num_linhas*num_colunas):
                fig.delaxes(axes.flatten()[i])
        
        
            # print(todos_indices_vertical)
        
            all_index15.extend(todos_indices_vertical)
        
            # Remover espaços vazios da lista
            all_index15 = list(filter(None, all_index15))
        
        
        print()
        print(satellite)
        
        
        # Substituir 'N' por 'Y' nos índices especificados
        df.loc[all_index15, 'outlier_flag'] = 'Y'
        
        
        
        
        
        
        
        #
        # Diretório de destino e diretório de saída desejado
        #destination_directory = "/media/debian-giorgio/DATA/GNSS_DATA/RINEX/PROCESSED"
        output_directory = os.path.join(str(ano), str(doy), estacao.upper())
        # #
        # # # Caminho completo do diretório de saída dentro do diretório de destino
        full_path = os.path.join(destination_directory)
        # #
        # #
        # # # Garantir que o diretório exista ou criar se não existir
        os.makedirs(full_path, exist_ok=True)
        # #
        # #
        # # # Definir o nome do arquivo
        file_name = f"{estacao}_{satellite}_{doy}_{ano}.RNX2"
        # #
        # # # Caminho completo do arquivo de saída
        output_file_path = os.path.join(full_path, file_name)
        # #
        # #
        # # # Selecionar apenas as colunas desejadas
        colunas_desejadas = ['date', 'time', 'mjd', 'pos_x', 'pos_y', 'pos_z', 'L1', 'L2', 'L5', 'P1', 'P2', 'P5', 'cs_flag', 'outlier_flag', 'satellite', 'sta', 'hght', 'El', 'Lon', 'Lat', 'obs_La', 'obs_Lb', 'obs_Lc', 'obs_Ca', 'obs_Cb', 'obs_Cc']
        
        
        
        df_selecionado = df[colunas_desejadas]
        # #
        # #
        # Substituir NaN por -999999.999
        df_selecionado = df_selecionado.fillna(-999999.999)
        
        # Salvar o DataFrame selecionado em um arquivo de texto separado por tabulação
        df_selecionado.to_csv(output_file_path, sep='\t', index=False, na_rep='-999999.999')
        
        # # Ler o arquivo para verificar se está correto
        # with open(output_file_path, 'r') as f:
        #     print(f.read())
        
        
        
        # Ler o arquivo para verificar se está correto
        with open(output_file_path, 'r') as f:
            file_content = f.read()
        
        print(f"Dados exportados para {output_file_path}.")




