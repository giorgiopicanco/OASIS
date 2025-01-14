
#!/bin/bash

# Função para exibir a versão do pacote
show_version() {
    echo "$1:"
    pip show "$1" | grep -E '^Version:'
}

# Exibir a versão de cada pacote
show_version datetime
show_version numpy
show_version pandas
show_version matplotlib
show_version astropy
show_version scipy
show_version scikit-learn
show_version georinex
show_version pyproj

