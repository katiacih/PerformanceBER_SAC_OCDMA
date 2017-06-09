#!/usr/bin/python2.7
#!-*- coding: utf8 -*-
#-*-coding:latin1-*-

from decimal import Decimal
import numpy as np
from math import log1p,sqrt,erfc
import numpy as np
import matplotlib.pyplot as plt


from array import array

# from Codigos.codeMS import *
# import "/Codigos/codeMS" as cody
from Codigos import codeMS, Ruidos
from paramiko.ber import BER



'''
Created on 16 de mai de 2017
Insira aqui testes 
@author: katia
'''

def testegrafico():
    
    plt.subplots_adjust(hspace=0.4)
    t = np.arange(0.01, 20.0, 0.01)
    
    # log y axis
    plt.subplot(221)
    plt.semilogy(t, np.exp(-t/5.0))
    plt.title('semilogy')
    plt.grid(True)
    
    # log x axis
    plt.subplot(222)
    plt.semilogx(t, np.sin(2*np.pi*t))
    plt.title('semilogx')
    plt.grid(True)
    
    # log x and y axis
    plt.subplot(223)
    plt.loglog(t, 20*np.exp(-t/10.0), basex=2)
    plt.grid(True)
    plt.title('log base 2 on x')
    
    # with errorbars: clip non-positive values
    ax = plt.subplot(224)
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    
    x = 10.0**np.linspace(0.0, 2.0, 20)
    y = x**2.0
    plt.errorbar(x, y, xerr=0.1*x, yerr=5.0 + 0.75*y)
    ax.set_ylim(ymin=0.1)
    ax.set_title('Errorbars go negative')
    
    
    plt.show()


def teste_code_MS():

    #definicoes de ruidos
     # dados
    n_eta = 0.6
    
    e = 1.6e-19
    Tn = 300.00
    h = 6.66e-34
    kb = 1.38e-23
    B = 311.00
    lambdaComprimentoOnda = 1550

    #frequencia central do
    v0 =3e8
    v0_tera = 194

    deltaV_Heartz = 3.75e+12
    deltaV = 3.75

    #calculo da responsividade
    R = Ruidos.calculo_R(n_eta,e,h,v0)

    RL=1030

    #Conversoes de Psr para teste
    array_Psr_DBM = [-35,-30,-25,-20,-15,-10,-5,0]
    array_Psr_KiloWatts = [3.16e-10,1e-9,3.162e-9,1e-8,3.1623e-8,1e-7,3.16228e-7,0.000001]
    array_Psr_DeciWatts = [-65,-60,-55,-50,-45,-40,-35,30]
    #usou psr em watts
    array_Psr_Watts = [0,0.000001,0.000003162278,0.00001,0.000031622777,0.0001,0.000316,0.001]

    array_Psr_MiliWatts = [0.000316227766,0.001,0.00316227766,0.63095734448,0.01,0.031622776602,0.1,0.316227766017,1]  
    array_Psr_HPE = [4.24e-10,1.34e-9,4.239e-9,1.3405e-8,4.239e-8,1.34048e-7,4.23898e-7,0.000001340483]


def formulacao_testes_MS():
    # definicoes de ruidos
    # dados

    e = 1.6e-19
    Tn = 300.00
    h = 6.66e-34
    kb = 1.38e-23
    B = 311.00
    lambdaComprimentoOnda = 1550

    deltaV_Hertz = 3.75e+14
    deltaV = 3.75
    deltaVTest = 6.22e+8

    RL = 1030
    n_eta = 0.6
    v0 = 3e+8
    # v0 TeraHeartz
    v0_Th = 1.94
    v0_h =1.94e+12
    v0_capiroto = 2.633e+22
    v0_nano =1.94E+21

    # calculo da responsividade
    R = Ruidos.calculo_R(n_eta, e, h,v0_capiroto)

    R_set = 753.0
    # 750 m microAmpers
    R_set2 = 5.00E-9


    #gerando o codigo
    #array_code = codeMS.generated_sequence_code_Ms(NB,N,W)


    #gera valores nao computaveis
    psr_dBm = -10
    psr_watts = Ruidos.converter_psr_DBM_watss(-10.0)

    psr_mili = 0.1
    psr_watts = 0.0001
    # chega proximo do esperado
    psr_mega = 1e-10
    psr_kiloW = 1e-7
    psr_Deciwatts = -40
    psr_horse_mecanico = 1.34102E-4
    psr_horse_eletric = 1.340448E-7

    # considerando o numero de usuarios de 0 a 100 com NB =2 e NB =3


    W = 4
    # Com o numero de usuarios variante entre 10 e 100 e NB =2 e NB =3
    #test_num_user(R, psr_dBm, B, W, e, deltaVTest, kb, Tn, RL)

    # Com o psr variando de -24 a 0dBM com NB=2  e NB =3

    v0_capiroto = 2.633e+22
    v0_nano = 1.94E+21

    # calculo da responsividade
    R = Ruidos.calculo_R(n_eta, e, h, v0_capiroto)

    R_set = 753.0
    # 750 m microAmpers
    R_set2 = 5.00E-9

    test_psr(R, -24, B, W, e, deltaV_Hertz, kb, Tn, RL)


def test_num_user(R,psr,B,W,e, deltaV, kb, Tn, RL):


    BER_MS_NB_2 = []
    BER_MS_NB_3 = []

    array_Numero_User = []
    W = 4
    n_users = 10
    LB_NB_2 = float(codeMS.calculoLB(W, 2))
    LB_NB_3 = float(codeMS.calculoLB(W, 3))

    while n_users <= 100:
        array_Numero_User.append(n_users)

        L_NB_2 = codeMS.calculoL(n_users, 2, LB_NB_2)
        L_NB_3 = codeMS.calculoL(n_users, 3, LB_NB_3)

        # BER_MS_1.append(  log1p(Ruidos.calculo_BER_ARtigo_MS(R, Psr, W, n_users, e, deltaV_Heartz, kb, Tn, B, RL,L,LB,NB)) )
        BER_MS_NB_2.append(
            Ruidos.calculo_BER_ARtigo_MS(R, psr, W, n_users, e, deltaV, kb, Tn, B, RL, L_NB_2, LB_NB_2, 2))
        BER_MS_NB_3.append(
            Ruidos.calculo_BER_ARtigo_MS(R, psr, W, n_users, e, deltaV, kb, Tn, B, RL, L_NB_3, LB_NB_3, 3))
        n_users = n_users + 10
    print "Analise considerando numero de usuarios de 10 a 100"
    print "NB = 2"
    print  BER_MS_NB_2
    print "NB = 3"
    print  BER_MS_NB_3



def test_psr(R,psr,B,W,e, deltaV, kb, Tn, RL):
    BER_MS_NB_2 = []
    BER_MS_NB_3 = []
    BER_MS_NB_4 = []

    array_psr = []
    W = 4
    n_users = 30
    LB_NB_2 = float(codeMS.calculoLB(W, 2))
    LB_NB_3 = float(codeMS.calculoLB(W, 3))
    LB_NB_4 = float(codeMS.calculoLB(W, 4))
    #calculando comprimento dos codigos para NB =2 e NB=3
    L_NB_2 = codeMS.calculoL(n_users, 2, LB_NB_2)
    L_NB_3 = codeMS.calculoL(n_users, 3, LB_NB_3)
    L_NB_4 = codeMS.calculoL(n_users, 4, LB_NB_4)

    while psr <= 0:
        array_psr.append(psr)
        # BER_MS_1.append(  log1p(Ruidos.calculo_BER_ARtigo_MS(R, Psr, W, n_users, e, deltaV_Heartz, kb, Tn, B, RL,L,LB,NB)) )
        BER_MS_NB_2.append(
            Ruidos.calculo_BER_ARtigo_MS(R, psr, W, n_users, e, deltaV, kb, Tn, B, RL, L_NB_2, LB_NB_2, 2))
        BER_MS_NB_3.append(
            Ruidos.calculo_BER_ARtigo_MS(R, psr, W, n_users, e, deltaV, kb, Tn, B, RL, L_NB_3, LB_NB_3, 3))
        BER_MS_NB_4.append(
            Ruidos.calculo_BER_ARtigo_MS(R, psr, W, n_users, e, deltaV, kb, Tn, B, RL, L_NB_4, LB_NB_4, 4))
        psr = psr + 4
    print array_psr
    print  BER_MS_NB_2
    print  BER_MS_NB_3
    print  BER_MS_NB_4

