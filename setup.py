#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:30:35 2022

@author: philipwallace
"""
import subprocess as sp 
import shlex

if __name__== '__main__':
    
    
    #sp.run(shlex.split('conda env create -f ./Set_Up/environment.yml'))
    
    #sp.run(shlex.split('conda activate raf-sam-kelvin'))
    
    
    sp.run(shlex.split('pip install --upgrade pip'))
    
    sp.run(shlex.split('pip install -r ./Set_Up/docker_requirements.txt'))
    
    #sp.run(shlex.split('conda install -c anaconda -n raf-sam-kelvin pyqt qt cvxopt==1.2.6'))
    
    #sp.run(shlex.split('pip install pyqt5'))
    
    #sp.run(shlex.split('python GUI/Good_Layout.py'))