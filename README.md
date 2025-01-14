# OASIS: Open-Access System for Ionospheric Studies
This repository contains the OASIS software, designed to automate the processing of GNSS data. The script is a powerful tool for researchers working with large datasets of GNSS observations and ionospheric indices such as DTEC, ROTI, and SIDX.

Key Features

    Batch Processing: Handles the analysis of multiple input files efficiently.
    Automation: Simplifies complex steps in the ionospheric data processing workflow.
    Flexibility: Allows configuration of variables such as input/output directories, GNSS station lists, and time intervals.
    Integration: Compatible with scientific tools for visualization and ionospheric modeling.

Requirements

    Bash shell (tested on Linux/Unix)
    GNU Parallel for parallel processing
    GNSS and ionospheric analysis tools

How to Use

    Configure the input variables in the script, such as:
        Input and output directories
        GNSS station list
        Time interval for processing
    Run the script in a terminal using the command:

./RUN_OASIS.sh

Results will be saved in the specified output directory.

 Contributions are welcome! Feel free to open issues or submit pull requests to improve the script.
