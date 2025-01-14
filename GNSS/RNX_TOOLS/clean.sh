#!/usr/bin/env bash

# Definindo o caminho do diretório
doy="251"
year="2017"

echo $doy $year
# sta_list=("UDEC")

#echo "Processing $sta_list..."


#dir_clean="/home/debian-giorgio/OASIS/GNSS_DATA/RINEX/$year/$doy"
dir_clean="/media/debian-giorgio/DATA/GNSS_DATA/RINEX/$year/$doy"
#dir_sc="/home/debian-giorgio/OASIS/GNSS_DATA/RINEX/PROCESSED/$year/$doy"
dir_sc="/media/debian-giorgio/DATA/GNSS_DATA/RINEX/PROCESSED/$year/$doy"


# Verificando a existência do diretório
if [ ! -d "$dir_clean" ]; then
    echo "Diretório não encontrado: $dir_clean"
    exit 1
fi

# Criando uma lista de nomes de arquivos dentro do diretório
sta_list=($(find "$dir_clean" -maxdepth 1 -type f -printf "%f\n" | cut -c1-4 | tr '[:lower:]' '[:upper:]'))

# Imprimindo a lista
echo "${sta_list[@]}"
echo -e "\n"

# Imprimindo o número de arquivos e a lista de arquivos
num_files=${#sta_list[@]}
echo "$num_files RINEX files found for this date"


# # Inicializando o contador de pastas únicas com pelo menos 5 arquivos
# num_unique_folders_with_5_or_more_files=0
#
# # Lista para manter o controle das pastas já contadas
# processed_folders=()
#
#
# # Loop sobre cada estação na lista
# for sta in "${sta_list[@]}"; do
#     # Verificando se o diretório da estação existe dentro de dir_sc
#     if [ -d "$dir_sc/$sta" ]; then
#         # Obtendo o caminho completo da pasta
#         folder_path="$dir_sc/$sta"
#         # Verificando se a pasta já foi contada
#         if [[ ! " ${processed_folders[@]} " =~ " ${folder_path} " ]]; then
#             # Adicionando a pasta à lista de pastas processadas
#             processed_folders+=("$folder_path")
#             # Contando o número de arquivos na pasta
#             num_files=$(find "$folder_path" -maxdepth 1 -type f | wc -l)
#             # Verificando se a pasta contém pelo menos 5 arquivos
#             if [ "$num_files" -ge 5 ]; then
#                 # Incrementando o contador de pastas únicas com pelo menos 5 arquivos
#                 ((num_unique_folders_with_5_or_more_files++))
#                 # Removendo a pasta da lista original
#                 sta_list=(${sta_list[@]//$sta/})
#             fi
#         fi
#     fi
# done
#
# echo -e "\n"
# # Imprimindo o número de pastas únicas que contêm pelo menos 5 arquivos
# echo "Number of GNSS stations already processed: $num_unique_folders_with_5_or_more_files"
# echo -e "\n"
# # Imprimindo o tamanho da lista sta_list
# echo "Now processing ${#sta_list[@]} GNSS stations..."
#
# # sta_list=("BADG")
# echo "Cleaning..."
#
# # Função para processar uma estação
# process_station() {
#     local sta=$1
#     local doy=$2
#     local year=$3
#     ./RNX_CLEAN.py "${sta,,}" "$doy" "$year"
# }
#
# export -f process_station
#
# # Paralelizando o processamento das estações
# printf "%s\n" "${sta_list[@]}" | parallel process_station {} "$doy" "$year"



# Atribuindo a lista de pastas a uma variável
sta_list3=$(find "$dir_sc" -maxdepth 1 -type d -exec sh -c 'count=$(ls -1 "{}"/*RINEX_SCREENED* 2>/dev/null | wc -l); if [ $count -lt 10 ]; then echo -n "{} "; fi' \;)

# Removendo o "./" da lista de pastas
sta_list3="${folders//$dir_sc\//}"

# Criando uma lista de nomes de arquivos dentro do diretório
sta_list3=($(find "$dir_clean" -maxdepth 1 -type f -printf "%f\n" | cut -c1-4 | tr '[:lower:]' '[:upper:]'))


# Exibindo a lista de pastas
echo "$sta_list3"


# Contando o número total de pastas no diretório
total_folders=$(find "$dir_sc" -maxdepth 1 -type d | wc -l)

# Calculando o número de pastas menos o número de elementos em sta_list3
num_folders_minus_stalist3=$((total_folders - ${#sta_list3[@]}))


echo -e "\n"
# Imprimindo o número de pastas únicas que contêm pelo menos 10 arquivos
echo "Number of GNSS stations already screened: $num_folders_minus_stalist3"
echo -e "\n"
# Imprimindo o tamanho da lista sta_list
echo "Now processing ${#sta_list3[@]} GNSS stations..."


# sta_list3=("BADG")


echo "Screening..."

# Iterando sobre as estações
for sta in "${sta_list3[@]}"; do
    # Removendo espaços extras da estação
    sta=$(echo "$sta" | tr -d '[:space:]' | tr '[:lower:]' '[:upper:]')
    # Diretório da estação
    estacao_dir="$dir_sc/${sta}"

    echo $estacao_dir


    # Verificando se o diretório da estação existe
    if [ -d "$estacao_dir" ]; then
        echo "Screening station: $sta"

        # Encontrando todos os arquivos no formato especificado no diretório
        arquivos=($(find "$estacao_dir" -type f -name '????_???_???_????_IPP_filtered.txt'))

        # Verificando se foram encontrados arquivos
        if [ ${#arquivos[@]} -eq 0 ]; then
            echo "No files found in directory: $estacao_dir"
        else
            # Iterando sobre os arquivos encontrados
            for arquivo in "${arquivos[@]}"; do
                echo -e "\n"
                echo "Processing file: $arquivo"
                # Screening
                ./RNX_SCREENING.py "$arquivo"
            done
        fi
    else
        echo "Directory not found: $estacao_dir"
    fi
done



# Criando uma lista de nomes de arquivos dentro do diretório
sta_list=($(find "$dir_clean" -maxdepth 1 -type f -printf "%f\n" | cut -c1-4 | tr '[:lower:]' '[:upper:]'))

sta_list=("BADG")

echo "Levelling..."

# Iterando sobre as estações
for sta in "${sta_list[@]}"; do
    # Removendo espaços extras da estação
    sta=$(echo "$sta" | tr -d '[:space:]')
    # Diretório da estação
    estacao_dir="$dir_sc/${sta^^}"

    # Verificando se o diretório da estação existe
    if [ -d "$estacao_dir" ]; then
        echo "Levelling station: $sta"
        # Substitua o comentário abaixo pela chamada do seu script Python
        python3 RNX_LEVELLING.py "${sta^^}" "$doy" "$year"
    else
        echo "Directory not found: $estacao_dir"
    fi
done


