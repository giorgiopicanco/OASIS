#!/usr/bin/env python3
import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time
from scipy.constants import speed_of_light
from pyOASIS import gnss_freqs
from pyOASIS import levelling_settings
import pyOASIS

 #
 # OBSERVATION ARC DEFINITION:
 # --------------------------
 #   Maximum number of epochs in the RINEX data    3000
 #   Sampling interval for RINEX data                n. seconds
 #   Maximum gap before starting a new arc          180. seconds  OK
 #   Minimum number of observations in an arc        10           OK
 #   Handling of event flags in RINEX files     skip file
 #   If S1&S2 observations are both zero        skip all observations
 #   If P2 unavailable                          do not use C2


# Acessando as variáveis passadas como argumentos


def RNXlevelling(estacao,diretorio_principal):
    # estacao = sys.argv[1][36:40]  # O primeiro argumento é o nome do próprio script, então pegamos os argumentos a partir do segundo até o antepenúltimo\
    # diretorio_principal = sys.argv[1]  # O último argumento

    # Variáveis e Parâmetros
    h1 = 0
    n_horas = 24 #horas
    int1 = 120 #minutos
    
    # Accessing the frequencies of the GPS system
    gps_freqs = gnss_freqs.FREQUENCY[gnss_freqs.GPS]
    f1 = gps_freqs[1]
    f2 = gps_freqs[2]
    f5 = gps_freqs[5]
    
    # Definir o nome do arquivo
    file_name = pyOASIS.__path__[0]+'/glonass_channels.dat'  #colocar opção sem barras
    
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
    
    
    
    
    # Construindo o caminho completo para a pasta CHPI
    caminho_ = os.path.join(diretorio_principal)
    
    # Verificando se o diretório CHPI existe
    if os.path.exists(caminho_):
        # Listando o conteúdo da pasta CHPI
        conteudo_ = os.listdir(caminho_)
        print("Arquivos .txt na pasta:")
    
        # # Definindo a variável arquivos de acordo com o conteúdo_chpi
        # arquivos = [arquivo for arquivo in conteudo_ if arquivo.startswith("RINEX_SCREENED_" + estacao) and arquivo.endswith(".txt")]
    
        # Definindo a variável arquivos para incluir apenas os arquivos que terminam com .RNX2
        arquivos = [arquivo for arquivo in conteudo_ if arquivo.endswith(".RNX2")]
    
        first = arquivos[0]
    
        doy = first[9:12]
        ano = first[13:17]
    
    
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
    cs_flag = []
    outlier_flag = []
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
    
    
    for arquivo in arquivos:
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
                    'cs_flag': colunas[12],
                    'outlier_flag': colunas[13],
                    'satellite': colunas[14],
                    'sta': colunas[15],
                    'hght': colunas[16],
                    'El': colunas[17],
                    'Lon': colunas[18],
                    'Lat': colunas[19],
                    'obs_La': colunas[20],
                    'obs_Lb': colunas[21],
                    'obs_Lc': colunas[22],
                    'obs_Ca': colunas[23],
                    'obs_Cb': colunas[24],
                    'obs_Cc': colunas[25]
                }
    
                # Adicionando os valores de cada variável às respectivas listas
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
                cs_flag.append(registro['cs_flag'])
                outlier_flag.append(registro['outlier_flag'])
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
    
    
    
    
    
    
    sat_classes = ['G','R']
    #sat_classes = ['R','G']
    #sat_classes = ['G']
    
    #plt.figure(figsize=(12, 6))
    # Iterando sobre os valores de sat_class
    for sat_class in sat_classes:
        sat = sat_class
    
        # Criar uma única figura
        plt.figure(figsize=(12, 6))
    
        # Filtrando os satélites
        if sat:
            satellites_to_plot = [sv for sv in np.unique(satellites) if sv.startswith(sat)]
        else:
            satellites_to_plot = np.unique(satellites)
    
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
        cs_flag_filtered = []
        outlier_flag_filtered = []
        satellite_filtered = []
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
    
    
        for satellite in satellites_to_plot:
            print(satellite)
    
            sat = satellite
    
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
    
            akl = 40.3 * 10 ** 16 * ((1 / f2 ** 2) - (1 / f1 ** 2))
    
            lambda1 = (speed_of_light / f1)
            lambda2 = (speed_of_light / f2)
            lambda5 = (speed_of_light / f5)
    
            # print(w1,w2)
    
    
            indices = np.where(np.array(satellites) == satellite)[0]
            # print(f"Índices para o satélite {satellite}")#  : {indices}")
    
            # Inicializando listas filtradas para cada satélite
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
            cs_flag_filtered = []
            outlier_flag_filtered = []
            satellite_filtered = []
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
                cs_flag_filtered.append(cs_flag[idx])
                outlier_flag_filtered.append(outlier_flag[idx])
                satellite_filtered.append(satellites[idx])
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
    
    
            # Construindo um DataFrame com as listas filtradas
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
                'P5': P5_filtered,
                'cs_flag': cs_flag_filtered,
                'outlier_flag': outlier_flag_filtered,
                'satellite': satellite_filtered,
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
    
            # # Convertendo a coluna 'timestamp' para o tipo datetime, se ainda não estiver
            # # df['timestamp'] = pd.to_datetime(df['timestamp'])
            #
            # # # Extraindo apenas o tempo de 'timestamp' e armazenando em uma nova coluna 'time'
            # df['time'] = df['timestamp'].dt.time
            #
            # # Convertendo a coluna 'timestamp' para o tipo datetime, se ainda não estiver
            # df['timestamp'] = pd.to_datetime(df['timestamp'])
            #
            # # Extraindo apenas o tempo de 'timestamp' e armazenando em uma nova coluna 'time'
            # df['time'] = df['timestamp'].dt.time
    
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
    
    
            # # Selecionar as linhas onde outlier_flag é igual a 'Y' e imprimir timestamp e cs_flag
            # print(df.loc[df['outlier_flag'] == 'Y', ['timestamp', 'cs_flag','outlier_flag']])
    
            # sys.exit()
    
            # Encontrar índices onde df['outlier_flag'] é 'Y'
            outlier_indices = df.index[df['outlier_flag'] == 'Y'].tolist()
    
            # Transformar essas posições em NaN para cada um dos arrays
            for index in outlier_indices:
                L1_array[index] = np.nan
                L2_array[index] = np.nan
                L5_array[index] = np.nan
                P1_array[index] = np.nan
                P2_array[index] = np.nan
                P5_array[index] = np.nan
    
    
            L_GF = levelling_settings.geometry_free_combination_L(lambda1, lambda2, L1_array, L2_array)
            P_GF = levelling_settings.geometry_free_combination_C(P1_array, P2_array)
    
            L_GF15 = levelling_settings.geometry_free_combination_L(lambda1, lambda5, L1_array, L5_array)
            P_GF15 = levelling_settings.geometry_free_combination_C(P1_array, P5_array)
    
            L_GF25 = levelling_settings.geometry_free_combination_L(lambda2, lambda5, L2_array, L5_array)
            P_GF25 = levelling_settings.geometry_free_combination_C(P2_array, P5_array)
    
    
    
            DE = 3
            Thr = DE + 1
            # arc_len = 200
            arc_len = 15
    
            # Obtendo os índices onde df['outlier_flag'] é igual a 'Y'
            indices_outliers = df.index[df['outlier_flag'] == 'Y']
    
            # Substituindo os valores em df['cs_flag'] por 'S' nos índices dos outliers
            df.loc[indices_outliers, 'cs_flag'] = 'S'
    
    
            # Suponho que df['cs_flag'] seja uma série do pandas
            arcos = []  # Lista para armazenar os arcos de observação
            arc_atual = []  # Lista temporária para armazenar o arco de observação atual
    
    
    
            # Criando a nova coluna 'mini_flag' preenchida com 'N'
            df['mini_flag'] = 'N'
    
            # Identificando onde 'outlier_flag' ou 'cs_flag' são 'S' ou 'Y' e substituindo 'N' em 'mini_flag' por 'Y'
            mask = (df['outlier_flag'] == 'S') | (df['outlier_flag'] == 'Y') | (df['cs_flag'] == 'S') | (df['cs_flag'] == 'Y')
            df.loc[mask, 'mini_flag'] = 'Y'
    
            # for i, (outlier, cs, mini) in enumerate(zip(df['outlier_flag'], df['cs_flag'], df['mini_flag'])):
            #     print(f"Índice: {i}, outlier_flag: {outlier}, cs_flag: {cs}, mini_flag: {mini}")
            #
            #
            # sys.exit()
    
            # Iterar sobre todos os elementos de df['cs_flag']
            for idx, value in enumerate(df['mini_flag']):
                if value == 'Y':
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
    
            # sys.exit()
    
            # Filtrar os arcos que passaram no critério de comprimento
            arcos = [arc for arc in arcos if len(df['cs_flag'][arc[0]:arc[-1]+1]) >= 15]
    
    
    
            # Passo 1: Criar uma cópia de L_GF
            L_GF2 = np.copy(L_GF)
    
            # Passo 2: Inicializar um array booleano com True
            marcador_fora_arco = np.ones_like(L_GF, dtype=bool)
    
            # Passo 3: Marcar os índices dentro dos arcos como False (não para transformar em NaN)
            # for start, end in arcos:
                # Iterar sobre cada arco válido e plotar os dados
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                marcador_fora_arco[start:end] = False
    
            # Passo 4: Transformar os valores fora dos arcos em NaN
            L_GF2[marcador_fora_arco] = np.nan
    
            # Atualizar L_GF para ser a versão modificada
            L_GF = L_GF2
    
    
    
            # Inicializar L_GF2 como uma cópia de L_GF
            L_GF2 = np.copy(L_GF)
    
            # Iterar pelos segmentos definidos em 'arcos'
            # for start, end in arcos:
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                # Selecionar o segmento atual de L_GF
                segmento_data = L_GF[start:end]
    
                # Calcular o primeiro e terceiro quartis
                Q1 = np.nanpercentile(segmento_data, 25)
                Q3 = np.nanpercentile(segmento_data, 75)
                IQR = Q3 - Q1
    
                # Definir limites para os valores atípicos
                lower_bound = Q1 - 8 * IQR
                upper_bound = Q3 + 8 * IQR
    
                # Encontrar índices dos valores atípicos dentro do segmento
                outlier_indices = np.where((segmento_data < lower_bound) | (segmento_data > upper_bound))
    
                # Substituir os outliers por NaN em L_GF2, não em segmento_data
                L_GF2[start:end][outlier_indices] = np.nan
    
            # Atualizar L_GF para ser a versão sem outliers
            L_GF = L_GF2
    
    
            # Primeiro, crie uma cópia de P_GF para manter os valores originais
            P_GF_adjusted = np.array(P_GF, copy=True)  # Supondo que P_GF é uma lista ou array NumPy
    
            # Inicialize todos os valores como NaN primeiro
            P_GF_adjusted[:] = np.nan
    
            # Agora, use um loop para definir valores de P_GF_adjusted para não-NaN dentro dos arcos
            # for start, end in arcos:
    
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                # Apenas copia os valores dentro do arco atual de P_GF para P_GF_adjusted
                P_GF_adjusted[start:end] = P_GF[start:end]
    
            # Agora, P_GF_adjusted terá NaN fora dos arcos definidos e manterá os valores originais dentro dos arcos
    
    
    
            L_GF_adjusted = list(L_GF)  # Inicializa com os valores originais
    
            # for start, end in arcos:
    
            # for i, arc in enumerate(arcos):
            #     start = arc[0]
            #     end = arc[-1]
            #     # Calcula a diferença e a média da diferença para o arco atual
            #     diff = P_GF_adjusted[start:end] - L_GF[start:end]
            #     mean_diff = np.nanmean(diff)
            #
            #     # Ajusta os valores de L_GF dentro deste arco
            #     for i in range(start, end):
            #         L_GF_adjusted[i] += mean_diff
    
    
            # Definindo cores para os arcos
            cores = ['blue', 'red', 'green', 'orange', 'purple']  # Adicione mais cores se necessário
    
    
    
    
    
            limiar = 2  # Defina o limiar adequado
    
            # Lista para armazenar os subarcos
            subarcos = []
    
            # Plotar os arcos originais na cor preta
            for i, arc in enumerate(arcos):
                L_GF_adjusted_arc = [L_GF_adjusted[i] for i in arc]
                P_GF_adjusted_arc = [P_GF[i] for i in arc]
    
    
                # # # plt.scatter(df['timestamp'][arc[0]:arc[-1]+1], L_GF_adjusted_arc, marker='o', color='black', s=100, label=f'Arco Original {i+1}')
                # # # plt.scatter(df['timestamp'][arc[0]:arc[-1]+1], P_GF_adjusted_arc, marker='o', color='green', s=100, label=f'Arco Original {i+1}')
    
    
    
            # Iterar sobre os arcos
            for arc in arcos:
                # Filtrar os valores de L_GF_adjusted correspondentes ao arco atual
                L_GF_adjusted_arc = [L_GF_adjusted[i] for i in arc]
    
                # Calcular a diferença entre pontos consecutivos
                valor = abs(np.diff(L_GF_adjusted_arc, prepend=np.nan))
    
                # Encontrar os índices onde a diferença ultrapassa o limiar
                outliers = np.where(valor > limiar)[0]
    
                # Se houver outliers, dividir o arco em subarcos
                if len(outliers) > 0:
                    subarco_indices = [arc[0]]  # Adicionar o índice inicial do arco
    
                    # Adicionar os índices dos outliers como divisores de subarco
                    for outlier in outliers:
                        subarco_indices.append(arc[outlier])  # Divisor de subarco
                        subarco_indices.append(arc[outlier] + 1)  # Próximo índice após o divisor
    
                    subarco_indices.append(arc[-1])  # Adicionar o índice final do arco
    
                    # Criar subarcos com os índices calculados
                    for i in range(0, len(subarco_indices), 2):
                        subarcos.append(list(range(subarco_indices[i], subarco_indices[i+1] + 1)))
                else:
                    # Se não houver outliers, manter o arco original
                    subarcos.append(arc)
    
            # # # # Plotar os valores de L_GF_adjusted para cada subarco
            # # # for i, subarco in enumerate(subarcos):
            # # #     L_GF_adjusted_subarco = [L_GF_adjusted[i] for i in subarco]
            # # #     plt.scatter(df['timestamp'][subarco[0]:subarco[-1]+1], L_GF_adjusted_subarco, marker='o', color=cores[i % len(cores)], label=f'Subarco {i+1}')
    
            # Adicionar o ponto outlier no gráfico (se necessário)
            # Suponha que 'outlier_index' seja o índice do outlier
            # plt.scatter(df['timestamp'][outlier_index], L_GF_adjusted[outlier_index], color='red', label='Outlier')
    
            # # # # Adicionar legendas ao gráfico
            # # # plt.legend()
            # # # plt.xlabel('Timestamp')
            # # # plt.ylabel('L_GF_adjusted')
            # # # plt.title('L_GF_adjusted para cada subarco')
            # # # plt.show()
    
            # Atualizar a lista de arcos para usar os subarcos
            arcos = subarcos
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            #############################################
            #############################################
            #############################################
            #############################################
            #############################################
            #############################################
            #############################################
            #############################################
            #############################################
            #############################################
    
    
    
    
            # Imprimir informações de cada arco e classificá-los
            for i, arc in enumerate(arcos):
                start_index = arc[0]
                end_index = arc[-1]
                num_observations = len(arc)
                status = "Mantido" if num_observations >= 15 else "Descartado"
                print(f"Arco {i + 1}: {df['timestamp'][start_index]} - {df['timestamp'][end_index]}, Start = {start_index}, End = {end_index}, "
                    f"Obs. = {num_observations}, Status = {status}")
    
    
            # Filtrar os arcos que passaram no critério de comprimento
            arcos = [arc for arc in arcos if len(df['cs_flag'][arc[0]:arc[-1]+1]) >= 15]
    
    
    
            # Passo 1: Criar uma cópia de L_GF
            L_GF2 = np.copy(L_GF)
    
            # Passo 2: Inicializar um array booleano com True
            marcador_fora_arco = np.ones_like(L_GF, dtype=bool)
    
            # Passo 3: Marcar os índices dentro dos arcos como False (não para transformar em NaN)
            # for start, end in arcos:
                # Iterar sobre cada arco válido e plotar os dados
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                marcador_fora_arco[start:end] = False
    
            # Passo 4: Transformar os valores fora dos arcos em NaN
            L_GF2[marcador_fora_arco] = np.nan
    
            # Atualizar L_GF para ser a versão modificada
            L_GF = L_GF2
    
    
    
            # Inicializar L_GF2 como uma cópia de L_GF
            L_GF2 = np.copy(L_GF)
    
            # Iterar pelos segmentos definidos em 'arcos'
            # for start, end in arcos:
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                # Selecionar o segmento atual de L_GF
                segmento_data = L_GF[start:end]
    
                # Calcular o primeiro e terceiro quartis
                Q1 = np.nanpercentile(segmento_data, 25)
                Q3 = np.nanpercentile(segmento_data, 75)
                IQR = Q3 - Q1
    
                # Definir limites para os valores atípicos
                lower_bound = Q1 - 8 * IQR
                upper_bound = Q3 + 8 * IQR
    
                # Encontrar índices dos valores atípicos dentro do segmento
                outlier_indices = np.where((segmento_data < lower_bound) | (segmento_data > upper_bound))
    
                # Substituir os outliers por NaN em L_GF2, não em segmento_data
                L_GF2[start:end][outlier_indices] = np.nan
    
            # Atualizar L_GF para ser a versão sem outliers
            L_GF = L_GF2
    
    
            # Primeiro, crie uma cópia de P_GF para manter os valores originais
            P_GF_adjusted = np.array(P_GF, copy=True)  # Supondo que P_GF é uma lista ou array NumPy
    
            # Inicialize todos os valores como NaN primeiro
            P_GF_adjusted[:] = np.nan
    
            # Agora, use um loop para definir valores de P_GF_adjusted para não-NaN dentro dos arcos
            # for start, end in arcos:
    
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                # Apenas copia os valores dentro do arco atual de P_GF para P_GF_adjusted
                P_GF_adjusted[start:end] = P_GF[start:end]
    
            # Agora, P_GF_adjusted terá NaN fora dos arcos definidos e manterá os valores originais dentro dos arcos
    
    
    
            L_GF_adjusted = list(L_GF_adjusted)  # Inicializa com os valores originais
    
    
    
    
    
    
    
            for arc in arcos:
                start = arc[0]
                end = arc[-1]
                print(start, end)
                # Calcula a diferença e a média da diferença para o arco atual
                diff = P_GF_adjusted[start:end] - L_GF_adjusted[start:end]
                mean_diff = np.nanmean(diff)
    
                # Ajusta os valores de L_GF dentro deste arco
                for j in range(start, end):
                    L_GF_adjusted[j] += mean_diff
    
    
            L_GF_adjusted_old = L_GF_adjusted
    
    
    
            # Definindo a função para remover outliers com base no método do quartil
            def remove_outliers_quartil(data):
                q1 = np.nanpercentile(data, 15)
                q3 = np.nanpercentile(data, 85)
                iqr = q3 - q1
                lower_bound = q1 - 1.3 * iqr
                upper_bound = q3 + 1.3 * iqr
                return [x for x in data if lower_bound <= x <= upper_bound]
    
    
            # Remover outliers individualmente para cada arco
            for arc in arcos:
                start = arc[0]
                end = arc[-1]
                L_GF_adjusted[start:end+1] = remove_outliers_quartil(L_GF_adjusted[start:end+1])
    
    
    
    
    
            L_GF_adjusted2 = L_GF_adjusted
            P_GF_adjusted2 = P_GF_adjusted
    
    
    
    
            # Suponha que L_GF e P_GF sejam suas séries de dados originais.
            # Inicialize as versões ajustadas com NaNs
            L_GF_adjusted = np.full_like(L_GF, np.nan, dtype=np.float64)
            P_GF_adjusted = np.full_like(P_GF, np.nan, dtype=np.float64)
    
            polynomial_fits = []
            polynomial_fits2 = []
            arc_data = []
            arc_data2 = []
    
            # Loop para o primeiro conjunto de arcos
            # for i, (start, end) in enumerate(arcos, start=1):
    
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                # Atualiza os valores de L_GF_adjusted e P_GF_adjusted apenas nos intervalos definidos pelos arcos
                L_GF_adjusted[start:end] = L_GF[start:end]
                P_GF_adjusted[start:end] = P_GF[start:end]
    
                arc_values = L_GF[start:end]  # Aqui, use os dados originais para ajuste
                arc_values2 = P_GF[start:end]  # Aqui, use os dados originais para ajuste
                arc_timestamps = df['timestamp'][start:end]
    
                # Ajustar um polinômio de segundo grau para cada conjunto de valores
                x_values = np.arange(len(arc_values))
                polynomial_fit = levelling_settings.fit_polynomial(x_values, arc_values, 3)  # Assumindo que esta função realiza o ajuste polinomial corretamente   #DIA=1.5
    
                x_values2 = np.arange(len(arc_values2))
                polynomial_fit2 = levelling_settings.fit_polynomial(x_values2, arc_values2, 3)  # Mesmo para o segundo conjunto de valores
    
                # Armazenar os ajustes e os dados dos arcos
                polynomial_fits.append(polynomial_fit)
                polynomial_fits2.append(polynomial_fit2)
                arc_data.append(arc_values)
                arc_data2.append(arc_values2)
    
    
    
            L_GF_adjusted = list(L_GF_adjusted)  # Inicializa com os valores originais
    
            # for start, end in arcos:
    
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                print(start,end)
                # Calcula a diferença e a média da diferença para o arco atual
                diff = P_GF_adjusted[start:end] - L_GF_adjusted[start:end]
                mean_diff = np.nanmean(diff)
    
                # Ajusta os valores de L_GF dentro deste arco
                for i in range(start, end):
                    L_GF_adjusted[i] += mean_diff
    
    
            print(len(L_GF_adjusted),len(L1_array))
    
    
    
    
    
    
            # Configurar pandas para exibir todas as colunas e linhas
            pd.set_option('display.max_columns', None)  # Substitui None pelo número de colunas se souber o número exato e preferir limitar
            pd.set_option('display.max_rows', None)  # Substitui None pelo número de linhas se souber o número exato e preferir limitar
            pd.set_option('display.max_colwidth', None)  # Ajusta a largura máxima das colunas para ver todo o conteúdo
            pd.set_option('display.width', None)  # Ajusta a largura da representação de saída para evitar quebra de linha automática
    
    
            # # Aplicar condições para definir valores fora do intervalo [-20, 20] como NaN
            # adjusted_array[(adjusted_array < -20) | (adjusted_array > 60)] = np.nan
    
            L_GF_adjusted2 = L_GF_adjusted
    
    
            # Calcular os quartis para todos os dados em L_GF_adjusted
            q1 = np.nanpercentile(L_GF_adjusted, 25)
            q3 = np.nanpercentile(L_GF_adjusted, 75)
            iqr = q3 - q1
    
            # Calcular os limites inferior e superior para identificar outliers
            lower_bound = q1 - 2.5 * iqr
            upper_bound = q3 + 7.5 * iqr
    
            # Transformar outliers em NaN considerando todos os dados
            L_GF_adjusted = [x if lower_bound <= x <= upper_bound else np.nan for x in L_GF_adjusted]
    
    
    
    
            #     # Preparação para plotar
            # # plt.figure(figsize=(12, 6))  # Definir o tamanho da figura
            # plt.plot(df['timestamp'], P_GF, marker='o', color='navy', linestyle='', linewidth=3, label='Dados Originais')
            # # plt.plot(df['timestamp'], L_GF_adjusted2, marker='o', color='red', linestyle='', linewidth=2, label='Dados Originais')
            # plt.plot(df['timestamp'], L_GF_adjusted, marker='o', color='red', linestyle='', linewidth=0.1, label='Dados Originais')
            plt.scatter(df['timestamp'], L_GF_adjusted, marker='o', s=20, color='blue', label='GF: L1-L2', zorder=0)
            # # plt.plot(df['timestamp'], [x / akl for x in L_GF_adjusted], marker='o', color='blue', linestyle='', linewidth=2, label='Dados Originais')
            # plt.show()
            #
    
    
    
            # # Definindo cores para os arcos
            # # cores = ['red', 'blue', 'green', 'orange', 'purple']  # Adicione mais cores se necessário
    
            cores = [
            'red', 'blue', 'green', 'orange', 'purple',
            'cyan', 'magenta', 'yellow', 'black', 'white',
            'gray', 'brown', 'pink', 'olive', 'navy',
            'teal', 'maroon', 'aquamarine', 'coral', 'crimson',
            'darkgreen', 'indigo', 'khaki', 'lavender', 'gold',
            'lime', 'tan', 'salmon', 'peru', 'orchid',
            'royalblue', 'seagreen', 'slategray', 'tomato', 'violet',
            'wheat', 'turquoise', 'thistle', 'steelblue', 'sienna',
            'sandybrown', 'rosybrown', 'powderblue', 'plum', 'pink',
            'palegreen', 'paleturquoise', 'palegoldenrod', 'palevioletred', 'papayawhip'
        ]
    
            #
            # # # Plotar cada arco separadamente com uma cor diferente
            # # for i, arc in enumerate(arcos):
            # #     start_index = arc[0]
            # #     end_index = arc[-1]
            # #     plt.plot(df['timestamp'][start_index:end_index+1], L_GF_adjusted[start_index:end_index+1], marker='o', color=cores[i % len(cores)], linestyle='', linewidth=2, label=f'Arco {i+1}')
    
    
    
            # Assegura que L_GF_adjusted tenha o mesmo comprimento que df
            if len(L_GF_adjusted) == len(df):
                df['LGF_combination'] = L_GF_adjusted
            else:
                print("Erro: L_GF_adjusted e df não têm o mesmo comprimento!")
    
    
    
    
    
    
    
    
            # ---------------------- [L1-L5]
    
    
    
            # Passo 1: Criar uma cópia de L_GF15
            L_GF3 = np.copy(L_GF15)
    
            # Passo 2: Inicializar um array booleano com True
            marcador_fora_arco15 = np.ones_like(L_GF15, dtype=bool)
    
            # Passo 3: Marcar os índices dentro dos arcos como False (não para transformar em NaN)
            # for start, end in arcos:
                # Iterar sobre cada arco válido e plotar os dados
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                marcador_fora_arco15[start:end] = False
    
            # Passo 4: Transformar os valores fora dos arcos em NaN
            L_GF3[marcador_fora_arco15] = np.nan
    
            # Atualizar L_GF para ser a versão modificada
            L_GF15 = L_GF3
    
    
    
            # Inicializar L_GF2 como uma cópia de L_GF
            L_GF3 = np.copy(L_GF15)
    
            # Iterar pelos segmentos definidos em 'arcos'
            # for start, end in arcos:
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                # Selecionar o segmento atual de L_GF
                segmento_data = L_GF3[start:end]
    
                # Calcular o primeiro e terceiro quartis
                Q1 = np.nanpercentile(segmento_data, 25)
                Q3 = np.nanpercentile(segmento_data, 75)
                IQR = Q3 - Q1
    
                # Definir limites para os valores atípicos
                lower_bound = Q1 - 8 * IQR
                upper_bound = Q3 + 8 * IQR
    
                # Encontrar índices dos valores atípicos dentro do segmento
                outlier_indices = np.where((segmento_data < lower_bound) | (segmento_data > upper_bound))
    
                # Substituir os outliers por NaN em L_GF2, não em segmento_data
                L_GF3[start:end][outlier_indices] = np.nan
    
            # Atualizar L_GF para ser a versão sem outliers
            L_GF15 = L_GF3
    
    
            # Primeiro, crie uma cópia de P_GF para manter os valores originais
            P_GF_adjusted15 = np.array(P_GF15, copy=True)  # Supondo que P_GF é uma lista ou array NumPy
    
            # Inicialize todos os valores como NaN primeiro
            P_GF_adjusted15[:] = np.nan
    
            # Agora, use um loop para definir valores de P_GF_adjusted para não-NaN dentro dos arcos
            # for start, end in arcos:
    
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                # Apenas copia os valores dentro do arco atual de P_GF para P_GF_adjusted
                P_GF_adjusted15[start:end] = P_GF15[start:end]
    
            # Agora, P_GF_adjusted terá NaN fora dos arcos definidos e manterá os valores originais dentro dos arcos
    
    
    
            L_GF_adjusted15 = list(L_GF15)  # Inicializa com os valores originais
    
            # for start, end in arcos:
    
            # for i, arc in enumerate(arcos):
            #     start = arc[0]
            #     end = arc[-1]
            #     # Calcula a diferença e a média da diferença para o arco atual
            #     diff = P_GF_adjusted[start:end] - L_GF[start:end]
            #     mean_diff = np.nanmean(diff)
            #
            #     # Ajusta os valores de L_GF dentro deste arco
            #     for i in range(start, end):
            #         L_GF_adjusted[i] += mean_diff
    
    
            # Definindo cores para os arcos
            cores = ['red', 'blue', 'green', 'orange', 'purple']  # Adicione mais cores se necessário
    
    
    
    
    
            limiar = 2  # Defina o limiar adequado
    
            # Lista para armazenar os subarcos
            subarcos = []
    
            # Plotar os arcos originais na cor preta
            for i, arc in enumerate(arcos):
                L_GF_adjusted_arc15 = [L_GF_adjusted15[i] for i in arc]
                P_GF_adjusted_arc15 = [P_GF15[i] for i in arc]
    
    
                # # # plt.scatter(df['timestamp'][arc[0]:arc[-1]+1], L_GF_adjusted_arc, marker='o', color='black', s=100, label=f'Arco Original {i+1}')
                # # # plt.scatter(df['timestamp'][arc[0]:arc[-1]+1], P_GF_adjusted_arc, marker='o', color='green', s=100, label=f'Arco Original {i+1}')
    
    
    
            # Iterar sobre os arcos
            for arc in arcos:
                # Filtrar os valores de L_GF_adjusted correspondentes ao arco atual
                L_GF_adjusted_arc15 = [L_GF_adjusted15[i] for i in arc]
    
                # Calcular a diferença entre pontos consecutivos
                valor = abs(np.diff(L_GF_adjusted_arc15, prepend=np.nan))
    
                # Encontrar os índices onde a diferença ultrapassa o limiar
                outliers = np.where(valor > limiar)[0]
    
                # Se houver outliers, dividir o arco em subarcos
                if len(outliers) > 0:
                    subarco_indices = [arc[0]]  # Adicionar o índice inicial do arco
    
                    # Adicionar os índices dos outliers como divisores de subarco
                    for outlier in outliers:
                        subarco_indices.append(arc[outlier])  # Divisor de subarco
                        subarco_indices.append(arc[outlier] + 1)  # Próximo índice após o divisor
    
                    subarco_indices.append(arc[-1])  # Adicionar o índice final do arco
    
                    # Criar subarcos com os índices calculados
                    for i in range(0, len(subarco_indices), 2):
                        subarcos.append(list(range(subarco_indices[i], subarco_indices[i+1] + 1)))
                else:
                    # Se não houver outliers, manter o arco original
                    subarcos.append(arc)
    
            # # # # Plotar os valores de L_GF_adjusted para cada subarco
            # # # for i, subarco in enumerate(subarcos):
            # # #     L_GF_adjusted_subarco = [L_GF_adjusted[i] for i in subarco]
            # # #     plt.scatter(df['timestamp'][subarco[0]:subarco[-1]+1], L_GF_adjusted_subarco, marker='o', color=cores[i % len(cores)], label=f'Subarco {i+1}')
    
            # Adicionar o ponto outlier no gráfico (se necessário)
            # Suponha que 'outlier_index' seja o índice do outlier
            # plt.scatter(df['timestamp'][outlier_index], L_GF_adjusted[outlier_index], color='red', label='Outlier')
    
            # # # # Adicionar legendas ao gráfico
            # # # plt.legend()
            # # # plt.xlabel('Timestamp')
            # # # plt.ylabel('L_GF_adjusted')
            # # # plt.title('L_GF_adjusted para cada subarco')
            # # # plt.show()
    
            # Atualizar a lista de arcos para usar os subarcos
            arcos = subarcos
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            #############################################
            #############################################
            #############################################
            #############################################
            #############################################
            #############################################
            #############################################
            #############################################
            #############################################
            #############################################
    
    
    
    
            # Imprimir informações de cada arco e classificá-los
            for i, arc in enumerate(arcos):
                start_index = arc[0]
                end_index = arc[-1]
                num_observations = len(arc)
                status = "Mantido" if num_observations >= 15 else "Descartado"
                print(f"Arco {i + 1}: {df['timestamp'][start_index]} - {df['timestamp'][end_index]}, Start = {start_index}, End = {end_index}, "
                    f"Obs. = {num_observations}, Status = {status}")
    
    
            # Filtrar os arcos que passaram no critério de comprimento
            arcos = [arc for arc in arcos if len(df['cs_flag'][arc[0]:arc[-1]+1]) >= 15]
    
    
    
            # Passo 1: Criar uma cópia de L_GF
            L_GF3 = np.copy(L_GF15)
    
            # Passo 2: Inicializar um array booleano com True
            marcador_fora_arco15 = np.ones_like(L_GF15, dtype=bool)
    
            # Passo 3: Marcar os índices dentro dos arcos como False (não para transformar em NaN)
            # for start, end in arcos:
                # Iterar sobre cada arco válido e plotar os dados
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                marcador_fora_arco15[start:end] = False
    
            # Passo 4: Transformar os valores fora dos arcos em NaN
            L_GF3[marcador_fora_arco15] = np.nan
    
            # Atualizar L_GF para ser a versão modificada
            L_GF15 = L_GF3
    
    
    
            # Inicializar L_GF2 como uma cópia de L_GF
            L_GF3 = np.copy(L_GF15)
    
            # Iterar pelos segmentos definidos em 'arcos'
            # for start, end in arcos:
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                # Selecionar o segmento atual de L_GF
                segmento_data = L_GF15[start:end]
    
                # Calcular o primeiro e terceiro quartis
                Q1 = np.nanpercentile(segmento_data, 25)
                Q3 = np.nanpercentile(segmento_data, 75)
                IQR = Q3 - Q1
    
                # Definir limites para os valores atípicos
                lower_bound = Q1 - 8 * IQR
                upper_bound = Q3 + 8 * IQR
    
                # Encontrar índices dos valores atípicos dentro do segmento
                outlier_indices = np.where((segmento_data < lower_bound) | (segmento_data > upper_bound))
    
                # Substituir os outliers por NaN em L_GF2, não em segmento_data
                L_GF3[start:end][outlier_indices] = np.nan
    
            # Atualizar L_GF para ser a versão sem outliers
            L_GF15 = L_GF3
    
    
            # Primeiro, crie uma cópia de P_GF para manter os valores originais
            P_GF_adjusted15= np.array(P_GF15, copy=True)  # Supondo que P_GF é uma lista ou array NumPy
    
            # Inicialize todos os valores como NaN primeiro
            P_GF_adjusted15[:] = np.nan
    
            # Agora, use um loop para definir valores de P_GF_adjusted para não-NaN dentro dos arcos
            # for start, end in arcos:
    
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                # Apenas copia os valores dentro do arco atual de P_GF para P_GF_adjusted
                P_GF_adjusted15[start:end] = P_GF15[start:end]
    
            # Agora, P_GF_adjusted terá NaN fora dos arcos definidos e manterá os valores originais dentro dos arcos
    
    
    
            L_GF_adjusted15 = list(L_GF_adjusted15)  # Inicializa com os valores originais
    
    
    
    
    
    
    
            for arc in arcos:
                start = arc[0]
                end = arc[-1]
                print(start, end)
                # Calcula a diferença e a média da diferença para o arco atual
                diff15 = P_GF_adjusted15[start:end] - L_GF_adjusted15[start:end]
                mean_diff15 = np.nanmean(diff15)
    
                # Ajusta os valores de L_GF dentro deste arco
                for j in range(start, end):
                    L_GF_adjusted15[j] += mean_diff15
    
    
            L_GF_adjusted_old15 = L_GF_adjusted15
    
    
    
            # Definindo a função para remover outliers com base no método do quartil
            def remove_outliers_quartil(data):
                q1 = np.nanpercentile(data, 15)
                q3 = np.nanpercentile(data, 85)
                iqr = q3 - q1
                lower_bound = q1 - 1.3 * iqr
                upper_bound = q3 + 1.3 * iqr
                return [x for x in data if lower_bound <= x <= upper_bound]
    
    
            # Remover outliers individualmente para cada arco
            for arc in arcos:
                start = arc[0]
                end = arc[-1]
                L_GF_adjusted15[start:end+1] = remove_outliers_quartil(L_GF_adjusted15[start:end+1])
    
    
    
    
    
            L_GF_adjusted3 = L_GF_adjusted15
            P_GF_adjusted3 = P_GF_adjusted15
    
    
    
    
            # Suponha que L_GF e P_GF sejam suas séries de dados originais.
            # Inicialize as versões ajustadas com NaNs
            L_GF_adjusted15 = np.full_like(L_GF15, np.nan, dtype=np.float64)
            P_GF_adjusted15 = np.full_like(P_GF15, np.nan, dtype=np.float64)
    
            polynomial_fits = []
            polynomial_fits2 = []
            arc_data = []
            arc_data2 = []
    
            # Loop para o primeiro conjunto de arcos
            # for i, (start, end) in enumerate(arcos, start=1):
    
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                # Atualiza os valores de L_GF_adjusted e P_GF_adjusted apenas nos intervalos definidos pelos arcos
                L_GF_adjusted15[start:end] = L_GF15[start:end]
                P_GF_adjusted15[start:end] = P_GF15[start:end]
    
                arc_values = L_GF15[start:end]  # Aqui, use os dados originais para ajuste
                arc_values2 = P_GF15[start:end]  # Aqui, use os dados originais para ajuste
                arc_timestamps = df['timestamp'][start:end]
    
                # Ajustar um polinômio de segundo grau para cada conjunto de valores
                x_values = np.arange(len(arc_values))
                polynomial_fit = levelling_settings.fit_polynomial(x_values, arc_values, 3)  # Assumindo que esta função realiza o ajuste polinomial corretamente   #DIA=1.5
    
                x_values2 = np.arange(len(arc_values2))
                polynomial_fit2 = levelling_settings.fit_polynomial(x_values2, arc_values2, 3)  # Mesmo para o segundo conjunto de valores
    
                # Armazenar os ajustes e os dados dos arcos
                polynomial_fits.append(polynomial_fit)
                polynomial_fits2.append(polynomial_fit2)
                arc_data.append(arc_values)
                arc_data2.append(arc_values2)
    
    
    
            L_GF_adjusted15 = list(L_GF_adjusted15)  # Inicializa com os valores originais
    
            # for start, end in arcos:
    
            for i, arc in enumerate(arcos):
                start = arc[0]
                end = arc[-1]
                print(start,end)
                # Calcula a diferença e a média da diferença para o arco atual
                diff15 = P_GF_adjusted15[start:end] - L_GF_adjusted15[start:end]
                mean_diff15 = np.nanmean(diff15)
    
                # Ajusta os valores de L_GF dentro deste arco
                for i in range(start, end):
                    L_GF_adjusted15[i] += mean_diff15
    
    
            print(len(L_GF_adjusted15),len(L1_array))
    
    
    
    
    
    
            # Configurar pandas para exibir todas as colunas e linhas
            pd.set_option('display.max_columns', None)  # Substitui None pelo número de colunas se souber o número exato e preferir limitar
            pd.set_option('display.max_rows', None)  # Substitui None pelo número de linhas se souber o número exato e preferir limitar
            pd.set_option('display.max_colwidth', None)  # Ajusta a largura máxima das colunas para ver todo o conteúdo
            pd.set_option('display.width', None)  # Ajusta a largura da representação de saída para evitar quebra de linha automática
    
    
            # # Aplicar condições para definir valores fora do intervalo [-20, 20] como NaN
            # adjusted_array[(adjusted_array < -20) | (adjusted_array > 60)] = np.nan
    
            L_GF_adjusted3 = L_GF_adjusted15
    
    
            # Calcular os quartis para todos os dados em L_GF_adjusted
            q1 = np.nanpercentile(L_GF_adjusted15, 25)
            q3 = np.nanpercentile(L_GF_adjusted15, 75)
            iqr = q3 - q1
    
            # Calcular os limites inferior e superior para identificar outliers
            lower_bound = q1 - 2.5 * iqr
            upper_bound = q3 + 7.5 * iqr
    
            # Transformar outliers em NaN considerando todos os dados
            L_GF_adjusted15 = [x if lower_bound <= x <= upper_bound else np.nan for x in L_GF_adjusted15]
    
    
    
    
            #     # Preparação para plotar
            # # plt.figure(figsize=(12, 6))  # Definir o tamanho da figura
            # plt.plot(df['timestamp'], P_GF, marker='o', color='navy', linestyle='', linewidth=3, label='Dados Originais')
            # # plt.plot(df['timestamp'], L_GF_adjusted2, marker='o', color='red', linestyle='', linewidth=2, label='Dados Originais')
            # plt.plot(df['timestamp'], L_GF_adjusted15, marker='o', color='lime', linestyle='', linewidth=0.1, label='Dados Originais')
            plt.scatter(df['timestamp'], L_GF_adjusted15, marker='o', s=20, color='red', label='GF: L1-L5', zorder=1)
    
            # # plt.plot(df['timestamp'], [x / akl for x in L_GF_adjusted], marker='o', color='blue', linestyle='', linewidth=2, label='Dados Originais')
            # plt.show()
            #
    
    
    
            # # Definindo cores para os arcos
            # # cores = ['red', 'blue', 'green', 'orange', 'purple']  # Adicione mais cores se necessário
    
            cores = [
            'red', 'blue', 'green', 'orange', 'purple',
            'cyan', 'magenta', 'yellow', 'black', 'white',
            'gray', 'brown', 'pink', 'olive', 'navy',
            'teal', 'maroon', 'aquamarine', 'coral', 'crimson',
            'darkgreen', 'indigo', 'khaki', 'lavender', 'gold',
            'lime', 'tan', 'salmon', 'peru', 'orchid',
            'royalblue', 'seagreen', 'slategray', 'tomato', 'violet',
            'wheat', 'turquoise', 'thistle', 'steelblue', 'sienna',
            'sandybrown', 'rosybrown', 'powderblue', 'plum', 'pink',
            'palegreen', 'paleturquoise', 'palegoldenrod', 'palevioletred', 'papayawhip'
        ]
    
            #
            # # # Plotar cada arco separadamente com uma cor diferente
            # # for i, arc in enumerate(arcos):
            # #     start_index = arc[0]
            # #     end_index = arc[-1]
            # #     plt.plot(df['timestamp'][start_index:end_index+1], L_GF_adjusted[start_index:end_index+1], marker='o', color=cores[i % len(cores)], linestyle='', linewidth=2, label=f'Arco {i+1}')
    
    
    
            # Assegura que L_GF_adjusted tenha o mesmo comprimento que df
            if len(L_GF_adjusted15) == len(df):
                df['LGF_combination15'] = L_GF_adjusted15
            else:
                print("Erro: L_GF_adjusted15 e df não têm o mesmo comprimento!")
    
    
    
    
    
            ########################################
    
            # Agora df tem a coluna LGF_combination
            # colunas_desejadas = ['timestamp', 'mjd', 'LGF_combination', 'satellite', 'sta', 'obs_La', 'obs_Lb', 'obs_Ca', 'obs_Cb']
            # Lista das colunas desejadas
            colunas_desejadas = [
                'date',
                'time2',
                'mjd',
                'pos_x',
                'pos_y',
                'pos_z',
                'LGF_combination',
                'LGF_combination15',
                'satellite',
                'sta',
                'hght',
                'El',
                'Lon',
                'Lat',
                'obs_La',
                'obs_Lb',
                'obs_Lc',
                'obs_Ca',
                'obs_Cb',
                'obs_Cc'
            ]
            df_selecionado = df[colunas_desejadas]
    
    
            # Caminho e nome do arquivo
            destination_directory = "/media/debian-giorgio/DATA/GNSS_DATA/RINEX/PROCESSED"
            #diretorio_principal
            output_directory = os.path.join(str(ano), str(doy), estacao.upper())
            full_path = os.path.join(diretorio_principal)
            file_name = f"{estacao}_{satellite}_{doy}_{ano}.RNX3"
            fig_name = f"{estacao}_{doy}_{ano}.png"
            output_file_path = os.path.join(full_path, file_name)
            output_fig_path = os.path.join(full_path, fig_name)
    
            # Garantir que o diretório exista
            os.makedirs(full_path, exist_ok=True)
    
            # Salvar o DataFrame selecionado em um arquivo de texto separado por tabulação
            df_selecionado.to_csv(output_file_path, sep='\t', index=False, na_rep='-999999.999')
    
            print(f"Dados exportados para {output_file_path}.")
    
    
    
    
    
        plt.xlabel('Time (UT)')
        plt.ylabel('Levelled Geometry-Free')
        # plt.title('Valores Ajustados de L_GF')
    
    
    
        plt.grid(axis='both', linestyle='--', color='black', linewidth=0.5)
        hours_fmt = mdates.DateFormatter('%H')
        plt.gca().xaxis.set_major_formatter(hours_fmt)
        minute_locator = mdates.MinuteLocator(interval=int1)
        plt.gca().xaxis.set_major_locator(minute_locator)
        plt.tick_params(axis='both', which='major', labelsize=12)
        # plt.ylim(0, 25)
    
        # Definir o tempo de início e limite de tempo
        #hora_inicio = h1 * 240
        #start_time = np.datetime64(df['timestamp'].iloc[hora_inicio])
        #plt.xlim([start_time, start_time + np.timedelta64(n_horas, 'h')])
    
        #
        # plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()







