# This code is to properly set up one's local machine to perform this anlaysis
# Need to have virtualenv and pip3 installed for Python3
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Special setup for certain packages
pip3 install git+https://github.com/compmonks/SOMPY.git
pip3 install ipdb==0.8.1