'''
Created on 17 de mai de 2017

@author: alunodisnel
'''
#!/usr/bin/python2.7
#!-*- coding: utf8 -*-
#-*-coding:latin1-*-

import numpy as np
from math import sqrt,erfc,sin ,pi,log10
import matplotlib.pyplot as plt
import scipy.special as sp
import sympy as sym




def sum_Cn_upper_arm(L,array_codes,W,n,code_class):
    sum = 0
    #n : indice do usuario a ser comparado
    for i in range(1,L+1):
        if (code_class.verify_same_sector(n,i) == True):
            if n == i:
                sum = sum  + W
            else:
                sum = sum + 1

    return sum
def sum_Cn_i_bit_a_bit(L,array_Cn):

    sum = 0
    #array_Cn : representa a sequencia de bits do usuario Cn
    for i in range(1,L+1):
        sum  = sum + array_Cn[i]


    return sum

def sum_Cn_lower_arm(L,array_codes,W,n,code_class):
    sum = 0
    #n : indice do usuario a ser comparado
    for i in range(1,L+1):
        if (code_class.verify_same_sector(n,i) == True):
            if n == i:
                sum = sum  + (code_class.NB -1)
            else:
                sum = sum + 1

    return sum

def sum_dl(n, NB,dn):
    sum = 0
    for i in range(n,NB+1):
        sum =sum+dn
    return sum

def converter_psr_DBM_watss(psr):
    return 1* (10 **(psr / 10)) / 1000

def Gv1_Psd1(psr,W,L,dl,dn,NB):
    G1_v_dv = ( (psr*W/L) * dl ) + ((psr/L)* sum_dl(1,NB,dn))
    return G1_v_dv

def Gv2_Psd2(psr,W,L,dl,dn,NB):
    G1_v_dv = ( (psr*(NB-1)/L) * dl ) + ((psr/L)* sum_dl(1,NB,dn))
    return G1_v_dv

def I_photocurrent(psr,W,L,dl,dn,NB,R):
    I1 = R * Gv1_Psd1(psr,W,L,dl,dn,NB)
    I2 = R * Gv2_Psd2(psr,W,L,dl,dn,NB)
    return I1-I2

def calc_uv(v):
    if v>=0:
        return 1
    else:
        return 0
# The properties of the proposed code at upper and lower arm of balanced receiver can be defined as
def calc_rv(v, v0, delta_v, L, NB, dn, psr,array_code):

    sum_Dn = 0
    for n in range(1, NB+1):

        sum_Cn = 0
        for i in range(1, L + 1):
            u = calc_uv(v)
            aux0 = v - v0 - ((delta_v / 2 * L) * (-L + (2 * i) - 2))
            aux1 = v - v0 - ((delta_v / 2 * L) * (-L + (2 * i)))
            function_cn = (u * aux0) - (u * aux1)
            sum_Cn = sum_Cn + function_cn
        function_Dn = dn + sum_Cn
        sum_Dn = sum_Dn + function_Dn
    return (psr / delta_v) * sum_Dn


def calculo_R(n_eta,e,h,v0):

    R = (n_eta* e)/(h* v0)
    return R

def calculo_Comprimento_codigo_N_FCC ( w , k):
    #atributo N : comprimento do codigo
    N =  (w * k)-(k - 1)
    #print "N_FCC = %f " %N
    return N

def calculo_Comprimento_codigo_N_RD ( w , k):
    #atributo N : comprimento do codigo
    N = k + (2 * w) - 3
    #print "N_RD = %f " %N


    return N
def calculo_Comprimento_codigo_N_MD ( w , k):
    #atributo N : comprimento do codigo
    N = k * w
    #print "N_RD = %f " %N


    return N
def calculo_Comprimento_codigo_N_MDW (w , k):
    #atributo N : comprimento do codigo
    a1 = ( k * pi ) / 3.0

    p1 =  ( (8/3) * ( ( sin(a1))**2 ) )
    N = ( 3 * k ) + p1
    #print "N_MDW = %f " %N

    return N

def calculo_Comprimento_codigo_N_EDW ( w , k):
    a1 = ( k * pi ) / 3.0

    a2 = ((k + 1) * pi ) / 3.0

    a3 =  ((k + 2) * pi ) / 3.0

    p1 =  ( (4.0/3.0) * ( ( sin(a1))**2 ) ) * ( (8.0/3.0) * ( ( sin(a2))**2 ) )

    p2=  ( (4.0/3.0) * ( ( sin(a3))**2 ) )

    N = ( ( 2 * k) + p1 + p2 )

    #print "N_EDW = %f " %N

    
    return  N

def atualizaComprimentoCodigo_N(codigos,n_usuarios):
    for k, v in codigos.items():
        if k == 'FCC':
            v['N'] = calculo_Comprimento_codigo_N_FCC( v['W'] , n_usuarios)
        if k == 'EDW':

            v['N'] = calculo_Comprimento_codigo_N_EDW( v['W'] , n_usuarios)
        if k == 'RD':
            v['N'] = calculo_Comprimento_codigo_N_RD( v['W'] , n_usuarios)
        if k == 'MDW':
            v['N'] = calculo_Comprimento_codigo_N_MDW( v['W'] , n_usuarios)
        if k == 'MD':
            v['N'] = calculo_Comprimento_codigo_N_MD( v['W'] , n_usuarios)


# -------------------------------------- Funcoes auxiliares para Ruido IMD--------------------------
def calcula_Ps(N):
    return 1/N

def funcao_U3(N,b):
        return  (3.0/2.0) * ( (2.0 * b * N) - (2.0 * (b**2.0)) + (2.0*b) + (N**2.0) - N)

def somatorio_U3(N, n_usuarios):
    b = 1
    soma = 0
    while b != n_usuarios:
        soma += funcao_U3(N,b)
        b += 1
    return soma
def calculo_a3(Go, OIP3):
    return (-2.0/3.0 ) * ( 10.0 ** ( ( ( 3.0 * Go ) / 20.0 )- ( OIP3/10.0 )  ) )

# --------------------------------------    Ruidos --------------------------

def Ruido_Disparado(e,Psr,R,N,W,B):
    w = W + 3
    Ish = 2 * e * B * R  * (Psr/ N) * w
    #e1 = (2 * Psr * e * B * R * (W + 3))
    #Ish = (2 * Psr * e * B * R * (W + 3 )/N

    #return (Ish)**2
    return Ish

def Ruido_Termico(kb,Tn,B,RL):
    
    ith = ( ( 4 * kb * Tn * B) / (RL))
    return ith


def Ruido_PIIN(deltaV,Psr,R,N,K,W,B):

    #Ruido de Intensidade Induzida por Fase
    #conversao de teraheartz para megahear
    deltaV_Mega_Heartz = 3750000
    deltaV_Heartz = 3.75e+12
    deltaV = 3.75
    # verificar conversao
    w = W + 3.0 
    p1 = ( (Psr**2.0) * B * (R**2.0) * K * ( W * (w)) )
    #p1 = ( Psr * B * R * K * ( W * w ))
    #p2= deltaV_Mega_Heartz * (N**2)
    p2= deltaV_Heartz * (N**2)
    #p2 =  deltaV * (N**2)
    iPIIN = ( p1 / p2  )
    return iPIIN 

def Ruido_IMD(N,n_usuarios):
    # valores de Go, OIP3 e Ps foram definidos de acordo com o artigo [9] citado
    Go = 11
    OIP3 = 47
    Ps = calcula_Ps(N)
    somatorio = somatorio_U3(N,n_usuarios)
    a3 = calculo_a3(Go,OIP3)

    return ( ( 9 * (a3 ** 2 ) / ( 16 * ( N ** 2 ) ) ) * Ps * (somatorio ** 2) )



# ---------------------------- Calculo Potencia do Sinal ------------------------------------

def calculo_I(R,Psr,W,N,k,e):
    e1 = (R * Psr * (W - 1)) / N
    return  e1 
def Serie_Fourier(N_simbolos, Ts,k):
    soma = 0
    n=1
    # modulacao ofdm
    #falta determinar Cn , j e t no calculo
    while n <= k:
        fn = (n - 1)/k
        n += 1
    B = N_simbolos / Ts 
    t = 1/B


# ------------------------------  Calculo de Relacao Sinal Ruido  ----------------------------

def calculo_SNR (deltaV,e,kb,Tn,RL,N,W,R,Psr,n_usuarios,B):

        I_shot = Ruido_Disparado(e,Psr,R,N,W,B)
        I_th = Ruido_Termico(kb,Tn,B,RL)
        I_PIIN = Ruido_PIIN(deltaV,Psr,R,N,n_usuarios,W,B)
        I_IMD = Ruido_IMD(N,n_usuarios)

        I_fotocorrente = calculo_I(R,Psr,W,N,n_usuarios,e)

        i_ruidos = I_shot + I_th + I_PIIN + I_IMD
        
        #i_ruidos = I_shot + I_th + I_PIIN 

        return  ( ( I_fotocorrente ** 2 ) / (i_ruidos** 2))
        #return  ( ( I_fotocorrente ** 2 ) / i_ruidos )

def calculo_BER(snr):
    return ( 0.5 * erfc( sqrt( ( snr/8.0 ) ) ) )

#------------------------ artigo 10 ---------------------------
def calculo_SNR_ART10(R,Psr,w,N,e,deltaV,k,kb,Tn,B,RL):
    i = ((2 * R * Psr * w)/N )**2
    ish = (2 * e * B * w * Psr * R)/N
    pi = (k -1 + w)
    iPIIN = ( (B * (R**2) * Psr * w  * k * pi) / (2 * (N**2) * deltaV))
    return i/(ish + iPIIN + Ruido_Termico(kb, Tn, B, RL))


def calculo_BER_ARtigo_MS(R, Psr, w, N, e, deltaV, kb, Tn, B, RL, L, LB, NB):
    R_2 =  (R ** 2)
    Psr_2 = ( Psr ** 2 )
    a1_2 = (    (   w - NB + 1  ) ** 2  )
    L_2 = (L ** 2)

    I_fotocorrente = (  R_2 * Psr_2  * a1_2) / L_2

    a2 = (w + (3 * NB) - 3)

    ruido_balistico = (e * B * R * Psr * a2 ) / L


    ruido_PIIN = ((B * R_2 * Psr_2 * N * w) / (2 * L_2 * deltaV) ) * a2

    ruido_termico = Ruido_Termico(kb, Tn, B, RL)

    sum_ruidos = ruido_balistico + ruido_PIIN + ruido_termico

    snr = I_fotocorrente / sum_ruidos

    if (snr > 0):
        raiz_snr = sqrt((snr / 8.0))
        #snr_erfc = erfc(10)
        snr_erfc = sp.erfc(raiz_snr)
        BER = (0.5 * snr_erfc )
        log_BER = log10(BER)
        return log_BER

    else:
        return 0
def snr_MS(R, Psr, w, N, e, deltaV, kb, Tn, B, RL, L, LB, NB,I):
    R_2 = (R ** 2)
    Psr_2 = (Psr ** 2)
    a1_2 = ((w - NB + 1) ** 2)
    L_2 = (L ** 2)
    ruido_balistico = (e * B * R * Psr * a2) / L

    ruido_PIIN = ((B * R_2 * Psr_2 * N * w) / 2 * L_2 * deltaV) * a2

    ruido_termico = Ruido_Termico(kb, Tn, B, RL)

    return I/(ruido_balistico+ruido_PIIN+ruido_termico)

    
    
    
    