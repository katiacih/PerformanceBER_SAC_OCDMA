#!/usr/bin/python2.7
#!-*- coding: utf8 -*-
#-*-coding:latin1-*-


import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from array import array
from Code_class import Code_class

'''
Created on 16 de mai de 2017

@author: alunodisnel

'''

def calc_pw(W):
    array_pw = []
    antecedent =0
    for w in range(1,W+1):
        if w > 1:
            c = antecedent + (w-1)
            antecedent = c
            array_pw.append(c)
        else:
            antecedent = 1
            array_pw.append(1)
    return array_pw
   
def calculoLB(W,NB):
    sum_i1 = 0
    for i in range (1,int(W+1)):
        sum_i1 = sum_i1 + i
    sum_i2 =0
    min_W_NB = int(W-NB)
    for i in range (1,(min_W_NB+1)):
        sum_i2 = sum_i2 + i
    return sum_i1 - sum_i2

def calculoL(N,NB,LB):
    N = float(N)
    NB = float(NB)
    LB = float(LB)
    if N == 0 :
        return 1
    div = N/NB
    return div*LB
def equals_Sector(n,l,dictionary):

    for k, v in dictionary.iteritems():

        print k, v

def upper_arm (dict,W,NB,n):
    sum_upper_arm = 0
    for i in range (1,L+1):
        if equals_Sector(n, i, dict) == True:
            if n == i:
                sum_upper_arm = sum_upper_arm + W
            else:
                sum_upper_arm = sum_upper_arm + 1
    return sum_upper_arm

def lower_arm (dict,W,NB,n):
    sum_lower_arm = 0
    for i in range (1,L+1):
        if equals_Sector(n, i, dict) == True:
            if n == i:
                sum_lower_arm =  sum_lower_arm  + (NB -1)
            else:
                sum_lower_arm =  sum_lower_arm  + 1
    return  sum_lower_arm

def functionAutoCorrelation(matrix1,L,matrix2):
    lambda_sum = 0
    for i in range(1,L):
        j=i+1
        while j < L:
            lambda_sum =  lambda_sum  + np.correlate(matrix1[i], matrix1[j], 'valid')
            j = j+1
    return lambda_sum
        
            
    
def functionCrossCorrelate(matrix1,L,matrix2):
    lambda_sum = 0
    for i in range(1,L):
        j=i+1
        while j < L:
            lambda_sum =  lambda_sum  + np.correlate(matrix1[i], matrix2[j], 'valid')
            j = j+1
    return lambda_sum
#def position_lambdaA_1(array1,NB,W,pw,LB,array2):
    #retorna a primeira posicao da matriz basica cujo lambda é igual a 1
def CrossCorrelate(matrix,j,NB):
    sum = 0
    for a in range(0,NB):
        sum = sum + matrix[a][j]
    #if(sum<2):
    #    s = 'The value of sum is ' + repr(sum) + ', and j is ' + repr(j) + '...'
    #    print(s)
    return sum

def overlaping(matrix,i,j,NB,LB):
    for col in range(j+2,LB):
        if(i-1>=0 and col-1>=0 and i<NB and col<LB):
            matrix[i][col] = matrix[i-1][col-1]
    return matrix
          
    
def matrix_basic(NB,W,pw,LB):
    
    matrix = np.zeros((NB,LB))
    print matrix
    pivo_deslocamento = 0
    #primeira posicao
    while len(pw) > 0:
        matrix[0][pw[0]-1] = 1
        pw.pop(0)
    #count_usr representa o usuario atual
    count_usr=1
    #print NB
    while count_usr < NB:
        #busca por '1' nao sobrepostos
        #c : numero de lambdac =1 encontrados
        c=0
        j=0
        while(j<LB+1):
            
            if (CrossCorrelate(matrix,j,NB) < 2 and c < count_usr):
                
                matrix[count_usr][j]=1
                c = c+1
                
            if(c == count_usr):
                col = j
                j = LB+1
            j=j+1
        #s = 'send for overlapping function index for i' + repr(count_usr) + ', and j is ' + repr(col) + '...'
        #print(s)
        #deslocamento apartir do ultimo '1' nao sobrepostos equivalente ao usuario
        matrix= overlaping(matrix, count_usr, col, NB, LB)
        count_usr = count_usr+1
            
    
    
    return matrix
def generated_sequence_code_Ms(NB,N,W):
    #definicoes do codigo
    # NB numero maximo de usuarios na matrix basica
    NB=4
    # numero  de usuarios 
    N=8
    W=4
    pw = []
    LB = calculoLB(W, NB)
    pw = calc_pw(W)

    matrix_CB = matrix_basic(NB, W, pw, LB)
    #print matrix_CB
    x = Code_class(matrix_CB,LB,NB,N)
    # array_final = x.technique_mapping()

    array_code = x.capeta_tecnica_de_mapeamento()
    #result = x.verify_same_sector(4,5,array_code)

    return array_code
        
    

        

     
        
        
    