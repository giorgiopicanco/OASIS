#!/usr/bin/env bash

# Accessing arguments
sta=${1^^}           # First argument: station name
doy=$2           # Second argument: day of year
year=$3         # Third argument: year
output_rinex=$4    # Fourth argument: output directory
output_roti=$5     # Fifth argument: RINEX directory


# Definindo o caminho do diretório
#doy="251"
#year="2017"

dir_sc="$output_rinex/$sta"
dir_roti="$output_roti/$sta"


echo $sta $doy
echo $dir_sc
echo $dir_roti



# Verificando a existência do diretório
if [ ! -d "$dir_sc" ]; then
    echo "Diretório não encontrado: $dir_clean"
    exit 1
fi



echo "ROTI calculating..."


# Diretório da estação
estacao_dir="$dir_sc"



# Verificando se o diretório da estação existe
if [ -d "$estacao_dir" ]; then
    echo "Obtaining ROTI for $sta station..."
    # Substitua o comentário abaixo pela chamada do seu script Python
    python3 ROTI_CALC.py "${sta^^}" "$doy" "$year"
else
    echo "Directory not found: $estacao_dir"
fi


