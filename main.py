#!/usr/bin/python2.7
#!-*- coding: utf8 -*-
#-*-coding:latin1-*-

from decimal import Decimal
import numpy as np
from math import log10,sqrt,erfc
import numpy as np
import matplotlib.pyplot as plt

from array import array

from Codigos import codeMS, Ruidos
from paramiko.ber import BER
import auxiliar
import Codigos


'''
Created on 16 de mai de 201s7       

@author: alunodisnel

'''
def main():
    
    #define
    #NB: Numero maximo de usuarios na matriz básica
    #N: numero de usuario simultaneos
    #W ponderação do codigo
    NB = 4
    W = 4
    N=20

    test_code_MS = auxiliar.teste_2()
    ber = []
    #ber = auxiliar.teste_code_MS()

    print ber



main()
