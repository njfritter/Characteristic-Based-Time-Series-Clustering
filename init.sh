#!/bin/bash
# This code is to properly set up one's local machine to perform this anlaysis
# Will use a virtual environment (venv) for analysis + pip3 to install all needed packages
# Python 3 comes with venv and pip3 pre installed
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Special setup for certain packages
pip3 install git+https://github.com/compmonks/SOMPY.git
pip3 install ipdb==0.8.1

"""
# OPTIONAL: Download WESAD data via cURL (can take up to 15 minutes, so ask user if they just want to download file themselves)
# The data that will be analyzed is packaged in pickle files for each subject (and each > 1GB each)
# Uploading this data to Github is cumbersome and bad practice, so I combined the data myself and uploaded the finish product using combine_wesad_data.py
# If you would like to get the raw data to look at yourself enter 'yes' below
"""

# First method: use 'read' command (https://stackoverflow.com/questions/226703/how-do-i-prompt-for-yes-no-cancel-input-in-a-linux-shell-script)
while true; do
    read -p "Downloading WESAD dataset via cURL; this could take up to 15 minutes. Do you wish to download via cURL (y) or manually (n)?" yn
    case $yn in
        [Yy]* ) curl -O https://uni-siegen.sciebo.de/s/pYjSgfOVs6Ntahr/download --output data/WESAD.zip; break;;
        [Nn]* ) echo "Please visit https://uni-siegen.sciebo.de/s/pYjSgfOVs6Ntahr/download to manually download the data. Please save to the data repo in the root directory."; exit;;
        * ) echo "Please enter yes or no.";;
    esac
done
