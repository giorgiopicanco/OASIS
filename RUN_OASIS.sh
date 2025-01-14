#!/usr/bin/env bash
# =============================================================================
# RUN_OASIS: Automation Script for GNSS Data Processing
# =============================================================================
# Description:
# This script is part of the OASIS (Open-Access System for Ionospheric Studies)
# framework. It automates the processing of GNSS data, generating ionospheric
# indices such as DTEC, ROTI, and SIDX. Designed for researchers working with
# large GNSS datasets.
#
# Author: Giorgio Picanço
# Email: giorgiopicanco@gmail.com
# License: Custom License (See LICENSE file for details)
#          Attribution Required | Non-Commercial Use Only
#
# Citation:
# Please cite the associated paper when using this script:
# "[Title of the Paper]," published in 2025, by Picanço, G.A.S.
#
# Requirements:
# - Bash shell
# - GNU Parallel
# - GNSS data processing tools
#
# Usage:
# Modify the input/output directories, station list, and other variables
# before running the script. Then execute:
#   ./RUN_OASIS.sh
#
# =============================================================================

valid_date=false

while [ "$valid_date" = false ]; do
    # Prompt the user to enter the year, month, and day in a single line
    read -p "Enter the year, month, and day (YYYY MM DD): " year month day

    # Attempt to calculate the day of the year (doy)
    doy=$(date -d "$year-$month-$day" +%j 2>/dev/null)

    if [ $? -eq 0 ]; then
        valid_date=true
    else
        echo "Invalid date. Please enter a valid date."
    fi
done

# Print a new line
echo

# Get the current directory
current_dir=$(pwd)

# Define RINEX_DIR
# RINEX_DIR="$current_dir/INPUT/RINEX/$year/$doy"
RINEX_DIR="/media/debian-giorgio/DATA/GNSS_DATA/RINEX/$year/$doy"

# Define ORBITS_DIR
# ORBITS_DIR="$current_dir/INPUT/ORBITS/$year/$doy"
ORBITS_DIR="/home/debian-giorgio/OASIS/GNSS_DATA/MGEX_ORBITS/$year/$doy"

# Define OUTPUT_RINEX
OUTPUT_RINEX="$current_dir/OUTPUT/RINEX/$year/$doy"

# Define OUTPUT_ORBITS
OUTPUT_ORBITS="$current_dir/OUTPUT/ORBITS/$year/$doy"

# Define OUTPUT_ROTI
OUTPUT_ROTI="$current_dir/OUTPUT/INDICES/ROTI/$year/$doy"

# Define OUTPUT_DTEC
OUTPUT_DTEC="$current_dir/OUTPUT/INDICES/DTEC/$year/$doy"

# Define OUTPUT_SIDX
OUTPUT_SIDX="$current_dir/OUTPUT/INDICES/SIDX/$year/$doy"

# Create OUTPUT_RINEX if it doesn't exist
if [ ! -d "$OUTPUT_RINEX" ]; then
    echo "Creating directory: $OUTPUT_RINEX"
    mkdir -p "$OUTPUT_RINEX" || { echo "Failed to create directory: $OUTPUT_RINEX"; exit 1; }
fi

# Create OUTPUT_ORBITS if it doesn't exist
if [ ! -d "$OUTPUT_ORBITS" ]; then
    echo "Creating directory: $OUTPUT_ORBITS"
    mkdir -p "$OUTPUT_ORBITS" || { echo "Failed to create directory: $OUTPUT_ORBITS"; exit 1; }
fi

# Print the variables
echo "Date selected: $year-$month-$day, DOY: $doy"
echo "RINEX_DIR is set to: $RINEX_DIR"

# Prompt the user for processing options
echo
echo "Choose an option:"
echo "1. Process all RINEX files for the selected date"
echo "2. Select a specific station (enter the 4-letter code)"

read -p "Option (1 or 2): " option

# Check the selected option
if [ "$option" == "1" ]; then
    # Verify the existence of the directory
    if [ ! -d "$RINEX_DIR" ]; then
        echo "Directory not found: $RINEX_DIR"
        exit 1
    fi

    # Create a list of file names within the directory
    sta_list=($(find "$RINEX_DIR" -maxdepth 1 -type f -printf "%f\n" | cut -c1-4 | tr '[:lower:]' '[:upper:]'))

elif [ "$option" == "2" ]; then
    # Prompt the user to enter the station code
    read -p "Enter the 4-letter station code (e.g., ABCD): " station_code

    # Validate the station code (should be 4 uppercase letters)
    if [[ ! "$station_code" =~ ^[A-Z]{4}$ ]]; then
        echo "Invalid station code. Please enter a 4-letter uppercase code."
        exit 1
    fi

    # Set the station list to the specified station code
    sta_list=("$station_code")

else
    echo "Invalid option. Please choose 1 or 2."
    exit 1
fi

# Print the selected station(s)
echo
echo "Selected station(s):"
echo "${sta_list[@]}"
echo

# Printing the number of files and the list of files
num_files=${#sta_list[@]}
echo "$num_files RINEX files found for this date"

# Ask the user if they want to correct and level the GNSS observations and process orbits before continuing
echo
echo "Do you want to process the GNSS orbits for this date (necessary for cycle-slip and outlier correction)?"
echo "Please choose an option:"
echo "1. Yes"
echo "2. No"
echo

read -p "Option (1 or 2): " option

# Check the selected option
if [ "$option" == "1" ]; then
    echo "Proceeding to process orbits."
    # Proceed to the next steps here
elif [ "$option" == "2" ]; then
    echo "Program terminated. Data will not be corrected."
    exit 0
else
    echo "Invalid option. Please choose 1 or 2."
    exit 1
fi

echo "Creating GNSS tabular orbits for this date..."

# Executar o script Python com os argumentos
python "$current_dir/GNSS/ORBITS_PROCESSING/SP3_INTERPOLATE.py" "$year" "$doy" "$ORBITS_DIR" "$OUTPUT_ORBITS"

echo
echo "Now processing ${#sta_list[@]} GNSS stations..."

echo "Cleaning..."

process_station() {
    local sta=$1
    local doy=$2
    local year=$3
    local output_dir=$4
    local rinex_dir=$5
    local output_orbits=$6

    # Runs the Python script with the passed arguments
    ./RNX_CLEAN.py "${sta,,}" "$doy" "$year" "$output_dir" "$rinex_dir" "$output_orbits"
}

export -f process_station

cd $current_dir/GNSS/RNX_TOOLS/

# Parallelizing the processing of stations
printf "%s\n" "${sta_list[@]}" | parallel process_station {} "$doy" "$year" "$OUTPUT_RINEX" "$RINEX_DIR" "$OUTPUT_ORBITS"

# Returns to the original directory
cd $current_dir/GNSS/RNX_TOOLS/

# Assign the list of directory names that have at least 38 files ending with .RNX1 to a variable, separated by space
sta_list2=$(find "$OUTPUT_RINEX" -maxdepth 1 -type d -exec sh -c 'for dir; do count=$(ls -1 "$dir"/*.RNX2 2>/dev/null | wc -l); if [ "$count" -ge 30 ]; then basename "$dir"; fi; done' sh {} + | tr '\n' ' ')

# Display the list of directories with at least 30 .RNX2 files
echo
echo "GNSS Stations already screened:"
if [ -z "$sta_list2" ]; then
    echo "None"
else
    echo "$sta_list2"
fi
echo

# Convert sta_list to an array
IFS=' ' read -r -a sta_list_array <<< "${sta_list[@]}"

# Convert sta_list2 to an array
IFS=' ' read -r -a exclude_array <<< "$sta_list2"

# Filter the original list to remove items that are in sta_list2
filtered_list=""
for item in "${sta_list_array[@]}"; do
    excluded=false
    for exclude in "${exclude_array[@]}"; do
        if [[ "$item" == "$exclude" ]]; then
            excluded=true
            break
        fi
    done
    if [ "$excluded" = false ]; then
        filtered_list+="$item "
    fi
done

# Remove the trailing extra space
filtered_list="${filtered_list% }"

# Convert the filtered list back to an array
IFS=' ' read -r -a filtered_array <<< "$filtered_list"

# Count the number of elements in the filtered array
num_elements=${#filtered_array[@]}

# Print the message with the number of elements
echo "Now screening $num_elements GNSS stations..."
echo

# Display the list of directories with at least 38 .RNX2 files
echo
echo "GNSS Stations:"
if [ -z "$filtered_array" ]; then
    echo "None"
else
    echo "$filtered_array"
fi
echo

# Join the elements of filtered_array into a single string separated by spaces
filtered_string=$(printf "%s " "${filtered_array[@]}")

# Assign filtered_array to sta_list3 directly as an array
sta_list3=("${filtered_array[@]}")

echo "Screening (or jumping to next step)..."


# Function to process the screening of a station
process_screening() {
    local sta=$1
    local output_rinex=$2

    # Remove extra spaces from the station name
    sta=$(echo "$sta" | tr -d '[:space:]' | tr '[:lower:]' '[:upper:]')

    # Station directory
    sta_dir="$output_rinex/${sta}"

    echo "$sta_dir"

    # Check if the station directory exists
    if [ -d "$sta_dir" ]; then
        echo "Screening station: $sta"

        # Find all files in the specified format
        files=($(find "$sta_dir" -type f -name '????_???_???_????.RNX1'))

        # Check if any files were found
        if [ ${#files[@]} -eq 0 ]; then
            echo "No files found in directory: $sta_dir"
        else
            # Iterate over the found files and run the screening script
            for file in "${files[@]}"; do
                echo -e "\n"
                echo "Processing file: $file"
                ./RNX_SCREENING.py "$file" "$sta_dir"
            done
        fi
    else
        echo "Directory not found: $sta_dir"
    fi
}

# Export the function for use with GNU parallel
export -f process_screening

# Use parallel to process the screening of each station in parallel
printf "%s\n" "${sta_list3[@]}" | parallel process_screening {} "$OUTPUT_RINEX"


echo
# Create a list to store folder names
filtered_folders=()

# Iterate over each folder in the OUTPUT_RINEX directory
for dir in "$OUTPUT_RINEX"/*/; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        # Count the number of .RNX2 files in the folder
        file_count=$(find "$dir" -maxdepth 1 -type f -name '*.RNX2' | wc -l)

        # If there are at least 10 .RNX2 files, add to the list
        if [ "$file_count" -ge 10 ]; then
            # Remove the path and keep only the folder name
            folder_name=$(basename "$dir")
            filtered_folders+=("$folder_name")
        fi
    fi
done

# Create a string with the folder names separated by space
sta_list=$(printf "%s " "${filtered_folders[@]}")

# Remove the last extra space
sta_list=$(echo "$sta_list" | sed 's/ *$//')

# Display the list of folders
echo "GNSS stations to be leveled into .RNX3 files:"
echo "$sta_list"
echo


echo "Levelling..."

# Function to perform levelling for each station
process_levelling() {
    local sta=$1
    local output_rinex=$2
    local doy=$3
    local year=$4

    # Remove extra spaces from the station name
    sta=$(echo "$sta" | tr -d '[:space:]')

    # Station directory
    estacao_dir="$output_rinex/${sta^^}"

    # Check if the station directory exists
    if [ -d "$estacao_dir" ]; then
        echo "Levelling station: $sta"

        # Replace with the Python levelling script
        python3 RNX_LEVELLING.py "${sta^^}" "$doy" "$year" "$estacao_dir"
    else
        echo "Directory not found: $estacao_dir"
    fi
}

# Export the function for use with GNU parallel
export -f process_levelling

# Use parallel to process the levelling of each station in parallel
printf "%s\n" "${filtered_folders[@]}" | parallel process_levelling {} "$OUTPUT_RINEX" "$doy" "$year"


# Removing trash...
for sta in "${sta_list[@]}"; do
    # Removing extra spaces from the station name
    sta=$(echo "$sta" | tr -d '[:space:]' | tr '[:lower:]' '[:upper:]')
    # Station directory
    sta_dir="$OUTPUT_RINEX/${sta}"

    # Checking if the station directory exists
    if [ -d "$sta_dir" ]; then
        echo
        #echo "Removing unnecessary data on: $sta_dir ..."
        cd "$sta_dir"

        # Check if there are any files with the extension .RNX1 or .RNX2 before trying to remove them
        if ls *.RNX1 1> /dev/null 2>&1 || ls *.RNX2 1> /dev/null 2>&1; then
            echo "Removing unnecessary .RNX1 and .RNX2 files..."
            rm -r *.RNX1 *.RNX2
        fi

    else
        echo
        echo "Nothing to be done!"
    fi
done

# Prompt the user for processing options
echo
echo "Done! Your GNSS data has been leveled and corrected for cycle slips and outliers. You can now proceed to study the ionospheric response to various phenomena using OASIS tools. Please select the phenomenon for which you'd like to analyze the ionospheric response:"
echo
echo "1. Equatorial Plasma Bubbles / Irregularities (ROTI)"
echo "2. Travelling Ionospheric Disturbances (dTEC)"
echo "3. Solar Flares (SIDX)"
echo "4. All (ROTI, dTEC, SIDX)"
echo "5. Terminate processes and exit"
echo

# Define the paths to the scripts using the current directory
roti_script="$current_dir/GNSS/INDICES/ROTI/ROTI_CALC.py"
dtec_script="$current_dir/GNSS/INDICES/DTEC/DTEC_CALC.py"
sidx_script="$current_dir/GNSS/INDICES/SIDX/SIDX_CALC.py"

# Create a list of file names within the directory
sta_list=($(find "$RINEX_DIR" -maxdepth 1 -type f -printf "%f\n" | cut -c1-4 | tr '[:lower:]' '[:upper:]'))

# Read user input for the selected ionospheric tool
while true; do
    read -p "Enter your choice (1, 2, 3, 4, or 5): " choice
    if [[ "$choice" =~ ^[1-5]$ ]]; then
        break
    else
        echo "Invalid option. Please enter 1, 2, 3, 4, or 5."
    fi
done

# Function to create directories based on user choice
create_directories() {
    local choice=$1
    for sta in "${sta_list[@]}"; do
        if [ "$choice" -eq 1 ] || [ "$choice" -eq 4 ]; then
            # Create OUTPUT_ROTI directory for the specific station if it doesn't exist
            if [ ! -d "$OUTPUT_ROTI/$sta" ]; then
                echo "Creating ROTI directory for $sta"
                mkdir -p "$OUTPUT_ROTI/$sta" || { echo "Failed to create directory: $OUTPUT_ROTI/$sta"; exit 1; }
            fi
        fi

        if [ "$choice" -eq 2 ] || [ "$choice" -eq 4 ]; then
            # Create OUTPUT_DTEC directory for the specific station if it doesn't exist
            if [ ! -d "$OUTPUT_DTEC/$sta" ]; then
                echo "Creating DTEC directory for $sta"
                mkdir -p "$OUTPUT_DTEC/$sta" || { echo "Failed to create directory: $OUTPUT_DTEC/$sta"; exit 1; }
            fi
        fi

        if [ "$choice" -eq 3 ] || [ "$choice" -eq 4 ]; then
            # Create OUTPUT_SIDX directory for the specific station if it doesn't exist
            if [ ! -d "$OUTPUT_SIDX/$sta" ]; then
                echo "Creating SIDX directory for $sta"
                mkdir -p "$OUTPUT_SIDX/$sta" || { echo "Failed to create directory: $OUTPUT_SIDX/$sta"; exit 1; }
            fi
        fi
    done
}

# Call the function to create the necessary directories based on user choice
create_directories "$choice"

# Process user choice
if [ "$choice" -eq 1 ]; then
    # Define a function to execute ROTI for each station
    process_roti() {
        local sta=$1
        local doy=$2
        local year=$3
        local output_rinex=$4
        local output_roti=$5
        local current_dir=$6

        echo "Obtaining ROTI for station: ${sta^^}..."

        # Check if the ROTI script exists
        if [ -f "$current_dir/GNSS/INDICES/ROTI/ROTI_CALC.py" ]; then
            python3 "$current_dir/GNSS/INDICES/ROTI/ROTI_CALC.py" "${sta^^}" "$doy" "$year" "$output_rinex" "$output_roti"
        else
            echo "Error: ROTI_CALC.py not found for station ${sta^^}!"
        fi
    }

    # Export the function so it can be used by parallel
    export -f process_roti

    # Use parallel to process each station in parallel
    printf "%s\n" "${sta_list[@]}" | parallel process_roti {} "$doy" "$year" "$OUTPUT_RINEX" "$OUTPUT_ROTI" "$current_dir"

elif [ "$choice" -eq 2 ]; then
    # Define a function to execute DTEC for each station
    process_dtec() {
        local sta=$1
        local doy=$2
        local year=$3
        local output_rinex=$4
        local output_dtec=$5
        local current_dir=$6

        echo "Obtaining DTEC for station: ${sta^^}..."

        # Check if the DTEC script exists
        if [ -f "$current_dir/GNSS/INDICES/DTEC/DTEC_CALC.py" ]; then
            python3 "$current_dir/GNSS/INDICES/DTEC/DTEC_CALC.py" "${sta^^}" "$doy" "$year" "$output_rinex" "$output_dtec"
        else
            echo "Error: DTEC_CALC.py not found for station ${sta^^}!"
        fi
    }

    # Export the function so it can be used by parallel
    export -f process_dtec

    # Use parallel to process each station in parallel
    printf "%s\n" "${sta_list[@]}" | parallel process_dtec {} "$doy" "$year" "$OUTPUT_RINEX" "$OUTPUT_DTEC" "$current_dir"

elif [ "$choice" -eq 3 ]; then
    # Define a function to execute SIDX for each station
    process_sidx() {
        local sta=$1
        local doy=$2
        local year=$3
        local output_rinex=$4
        local output_sidx=$5
        local current_dir=$6

        echo "Obtaining SIDX for station: ${sta^^}..."

        # Check if the SIDX script exists
        if [ -f "$current_dir/GNSS/INDICES/SIDX/SIDX_CALC.py" ]; then
            python3 "$current_dir/GNSS/INDICES/SIDX/SIDX_CALC.py" "${sta^^}" "$doy" "$year" "$output_rinex" "$output_sidx"
        else
            echo "Error: SIDX_CALC.py not found for station ${sta^^}!"
        fi
    }

    # Export the function so it can be used by parallel
    export -f process_sidx

    # Use parallel to process each station in parallel
    printf "%s\n" "${sta_list[@]}" | parallel process_sidx {} "$doy" "$year" "$OUTPUT_RINEX" "$OUTPUT_SIDX" "$current_dir"

elif [ "$choice" -eq 4 ]; then
    # Define uma função para executar todos os scripts (ROTI, DTEC, SIDX) para cada estação
    process_all_indices() {
        local sta=$1
        local doy=$2
        local year=$3
        local output_rinex=$4
        local output_roti=$5
        local output_dtec=$6
        local output_sidx=$7
        local current_dir=$8

        echo "Obtaining ROTI, DTEC, and SIDX for station: ${sta^^}..."

        # Executar ROTI
        if [ -f "$current_dir/GNSS/INDICES/ROTI/ROTI_CALC.py" ]; then
            python3 "$current_dir/GNSS/INDICES/ROTI/ROTI_CALC.py" "${sta^^}" "$doy" "$year" "$output_rinex" "$output_roti"
        else
            echo "Error: ROTI_CALC.py not found for station ${sta^^}!"
        fi

        # Executar DTEC
        if [ -f "$current_dir/GNSS/INDICES/DTEC/DTEC_CALC.py" ]; then
            python3 "$current_dir/GNSS/INDICES/DTEC/DTEC_CALC.py" "${sta^^}" "$doy" "$year" "$output_rinex" "$output_dtec"
        else
            echo "Error: DTEC_CALC.py not found for station ${sta^^}!"
        fi

        # Executar SIDX
        if [ -f "$current_dir/GNSS/INDICES/SIDX/SIDX_CALC.py" ]; then
            python3 "$current_dir/GNSS/INDICES/SIDX/SIDX_CALC.py" "${sta^^}" "$doy" "$year" "$output_rinex" "$output_sidx"
        else
            echo "Error: SIDX_CALC.py not found for station ${sta^^}!"
        fi
    }

    # Exportar a função para usar no paralelo
    export -f process_all_indices

    # Usar parallel para processar cada estação em paralelo para todos os índices
    printf "%s\n" "${sta_list[@]}" | parallel process_all_indices {} "$doy" "$year" "$OUTPUT_RINEX" "$OUTPUT_ROTI" "$OUTPUT_DTEC" "$OUTPUT_SIDX" "$current_dir"

elif [ "$choice" -eq 5 ]; then
    echo "Terminating processes and exiting..."
    exit 0
fi #

