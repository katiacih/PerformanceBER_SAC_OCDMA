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


    # considerando o numero de usuarios de 0 a 100 com NB =2 e NB =3

    BER_MS_NB_2 = []
    BER_MS_NB_3 = []
    array_Numero_User =[]
    W =4
    n_users = 10
    LB_NB_2 = codeMS.calculoLB(W, 2)
    LB_NB_3 = codeMS.calculoLB(W, 3)
    Psr = -10
    while n_users <=100:
        
        array_Numero_User.append(n_users)
       
        L2=codeMS.calculoL(n_users, 2, LB_NB_2)
        L3=codeMS.calculoL(n_users, 3, LB_NB_3)
    
        #BER_MS_1.append(  log1p(Ruidos.calculo_BER_ARtigo_MS(R, Psr, W, n_users, e, deltaV_Heartz, kb, Tn, B, RL,L,LB,NB)) )
        BER_MS_NB_2.append( Ruidos.calculo_BER_ARtigo_MS(R, Psr, W, n_users, e, deltaV_Heartz, kb, Tn, B, RL,L2,LB_NB_2,2))
        BER_MS_NB_3.append(  Ruidos.calculo_BER_ARtigo_MS(R, Psr, W, n_users, e, deltaV_Heartz, kb, Tn, B, RL,L3,LB_NB_3,3) )
        n_users = n_users+10
    print  BER_MS_NB_2
    print  BER_MS_NB_3
   
    '''
    #considerando taxas de transmissao diferentes com NB =2
    
  
    taxa_de_transmissao_1 = 1.25
    taxa_de_transmissao_2 = 2.5
    taxa_de_transmissao_3 = 5
    taxa_de_transmissao_4 = 10
    BER_MS_tx_1 = []
    BER_MS_tx_2 = []
    BER_MS_tx_3 = []
    BER_MS_tx_4 = []
    
    while n_users <=100:
    
        array_Numero_User.append(n_users)
        L=codeMS.calculoL(n_users, NB, LB)
        
        BER_MS_tx_1.append( log1p(Ruidos.calculo_BER_ARtigo_MS(R, Psr, W, n_users, e, deltaV_Heartz, kb, Tn, B, RL,L,LB,NB)) )
        BER_MS_tx_2.append( log1p(Ruidos.calculo_BER_ARtigo_MS(R, Psr, W, n_users, e, deltaV_Heartz, kb, Tn, B, RL,L,LB,NB)) )
        BER_MS_tx_3.append( log1p(Ruidos.calculo_BER_ARtigo_MS(R, Psr, W, n_users, e, deltaV_Heartz, kb, Tn, B, RL,L,LB,NB)) )
        BER_MS_tx_4.append( log1p(Ruidos.calculo_BER_ARtigo_MS(R, Psr, W, n_users, e, deltaV_Heartz, kb, Tn, B, RL,L,LB,NB)) )
                
        #BER_MS_1.append( Ruidos.calculo_BER_ARtigo_MS(R, Psr, W, n_users, e, deltaV_Heartz, kb, Tn, B, RL,L,LB,NB) )
        
        n_users = n_users+10

    # ------------- Graficos com diferentes ruidos
    plt.figure(1)
    #print BER_MS_NB_2
    
    plt.plot(array_Numero_User,BER_MS_NB_2,'go-',color ='green',label = u'$NB = 2 $ e $ W=4')
    plt.plot(array_Numero_User,BER_MS_NB_3,'go-',color ='red',label = u'$NB = 3 $ e $ W=4')
    
    plt.legend()
    
    minY = min(BER_MS_NB_2)
    maxY = max(BER_MS_NB_3)
    
    plt.yscale('symlog')
    
    #Add a comment to this line
    plt.title(u"Variacao de NB")
    plt.xlabel(u"N - Numero de Usuarios")
    plt.ylabel(u"Bit Error Rate - BER")
    
    plt.grid(True)
    
    plt.axis([0, 100, minY,maxY])
    
    #considerando Psr variando de -24 a 0
    L_NB_2 = codeMS.calculoLB(W, 2)
    L_NB_3 = codeMS.calculoLB(W, 3)
    L_NB_4 = codeMS.calculoLB(W, 4)
    BER_MS_2_psr = []
    BER_MS_3_psr = []
    BER_MS_4_psr = []
    psr =-24
    array_Psr = []
    n_users = 30
    #gerar array com psr em 
    while psr <=0:
        array_Psr.append(psr)
        
        BER_MS_2_psr.append( log1p(Ruidos.calculo_BER_ARtigo_MS(R, psr, W, n_users, e, deltaV_Heartz, kb, Tn, B, RL,L_NB_2,LB,2)) )
        BER_MS_3_psr.append( log1p(Ruidos.calculo_BER_ARtigo_MS(R,psr, W, n_users, e, deltaV_Heartz, kb, Tn, B, RL,L_NB_3,LB,3)) )
        BER_MS_4_psr.append( log1p(Ruidos.calculo_BER_ARtigo_MS(R, psr, W, n_users, e, deltaV_Heartz, kb, Tn, B, RL,L_NB_4,LB,4)) )
        
        psr = psr + 3
    
    plt.figure(2)
    
    plt.plot(array_Psr,BER_MS_2_psr,'go-',color ='green',label = u'$NB = 2 $ e $ W=4')
    plt.plot(array_Psr,BER_MS_3_psr,'go-',color ='red',label = u'$NB = 3 $ e $ W=4')
    plt.plot(array_Psr,BER_MS_4_psr,'go-',color ='blue',label = u'$NB = 4 $ e $ W=4')
    
    plt.legend()
    
    minY = min(BER_MS_2_psr)
    maxY = max(BER_MS_2_psr)
    
    plt.yscale('symlog')
    
    #Add a comment to this line
    plt.title(u"Variacao de psr ")
    plt.xlabel(u"Psr - Potencia do Receptor")
    plt.ylabel(u"Bit Error Rate - BER")
    
    plt.grid(True)
    
    plt.axis([0, 100, minY,maxY])
    
    plt.show()
    
    '''
def teste_2():
    # definicoes de ruidos
    # dados

    e = 1.6e-19
    Tn = 300.00
    h = 6.66e-34
    kb = 1.38e-23
    B = 311.00
    lambdaComprimentoOnda = 1550

    deltaV_Heartz = 3.75e+12
    deltaV = 3.75

    RL = 1030
    n_eta = 0.6
    v0 = 3 * 10 ** 8
    # v0 TeraHeartz
    v0_Th = 1.94
    v0_h =1.94e+12
    #
    v0_nano =1.94E+21

    # calculo da responsividade
    R = Ruidos.calculo_R(n_eta, e, h,v0_h)

    W = 4.0

    NB = 4.0
    N = 90.0
    LB_NB_4 = float(codeMS.calculoLB(W, NB))
    L =codeMS.calculoL(N,NB,LB_NB_4)

    array_code = codeMS.generated_sequence_code_Ms(NB,N,W)

    #gera valores nao computaveis
    psr_dBm = -10
    psr_watts = Ruidos.converter_psr_DBM_watss(-10.0)

    psr_mili = 0.1
    # chega proximo do esperado
    psr_mega = 1e-10
    psr_kiloW = 1e-7
    psr_Deciwatts = -40
    psr_horse_mecanico = 1.34102e-4
    psr_horse_eletric = 1.340448e-7


    #dn :  é o bit de dado de cada usuario
    # v =?
    '''
    #calculos variando de dl a dn de cada usuario
    
    rv:  PSD(densidade de potencia espctral) do receptor

    #calculo_rv = Ruidos.calc_rv(v, v0, deltaV_Heartz, L, NB, dn, psr,array_code)
    
    G1 : photodetector PD1 e photodetector PD2

    result_G1 = Ruidos.Gv1_Psd1(psr,W,L,dl,dn,NB)

    result_G2 = Ruidos.Gv2_Psd2(psr,W,L,dl,dn,NB)
    
    #calculo do fotodetector

    I = Ruidos.I_photocurrent(psr,W,L,dl,dn,NB,R)

    snr = Ruidos.snr_MS(R, psr, W, N, e, deltaV, kb, Tn, B, RL, L, LB_NB_4, NB,I)


    if (snr > 0):
        snr = sqrt((snr / 8.0))
        BER = (0.5 * erfc(snr))
        return BER
        
    '''
    n_users = 10
    array_Numero_User =[]
    array_BER_Numero_User =[]

    while n_users <= 100:
        array_Numero_User.append(n_users)
        L = codeMS.calculoL(n_users, NB, LB_NB_4)
        array_BER_Numero_User.append( Ruidos.calculo_BER_ARtigo_MS(R,psr_mega, W, n_users, e, deltaV_Heartz, kb, Tn, B, RL, L, LB_NB_4, NB))
        n_users = n_users +10
        


    #return Ruidos.calculo_BER_ARtigo_MS(R,psr_mega, W, N, e, deltaV_Heartz, kb, Tn, B, RL, L, LB_NB_4, NB)
    #return Ruidos.calculo_BER_ARtigo_MS(R,psr_Deciwatts, W, N, e, deltaV_Heartz, kb, Tn, B, RL, L, LB_NB_4, NB)
    #return Ruidos.calculo_BER_ARtigo_MS(R,psr_dBm, W, N, e, deltaV_Heartz, kb, Tn, B, RL, L, LB_NB_4, NB)
    return array_BER_Numero_User


