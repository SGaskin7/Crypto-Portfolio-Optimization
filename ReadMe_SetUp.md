# Set Up and Run Instructions 

This application requires Python 3.9+ and the Anaconda python package manager in order to install and manage the relevent dependencies. 

## Conda 

If you already have conda installed, you can skip this section.

Anaconda can be installed from its website with different versions depending on your operating system:

- [Anaconda for Windows](https://docs.anaconda.com/anaconda/install/windows/)
- [Anaconda for macOS](https://docs.anaconda.com/anaconda/install/mac-os/)
- [Anaconda for Linux](https://docs.anaconda.com/anaconda/install/linux/)

After following the directions, conda should be installed and functioning on your system.

## Installing our Repo

After downloading all the code and placing the repository in a location of your choice, open up a terminal window and navigate to the directory where the repo is located. 

Folders such as CVaR, Crypto Data and Factor Research should be visible. 

### Initialize a new virtual environment

Run these commands in terminal:

    conda env create -f ./Set_Up/environment.yml
    conda activate raf-sam-kelvin
    
To get the rest of the necessary packages, run:

    python setup.py
    
### Running the GUI

To run the GUI, run the command:

    python GUI/Good_Layout.py
