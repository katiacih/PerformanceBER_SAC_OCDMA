
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from array import array

class Code_class:
    """classe que ira lidar com sequencias genericas de codigo da forma:


                C1 [ array de bits de tamanho LB]
                C2 [array de bits]
                .
                .
                .
                CN []

    """
    matrix_basic = []
    matrix_zeros = []

    #inicializacoes
    def __init__(self, matrix_CB,LB,NB,N):
        self.matrix_basic = matrix_CB
        self.matrix_zeros = np.zeros((NB+1,LB), dtype=np.int)
        self.LB = LB
        self.NB = NB
        self.N = N


    #
    def join_array(self, array_CB,array_Z,NB,M):
        for i in range(len(array_CB)):
            for j in range(len(array_CB[i])):
                # iterando cada linha
                print(array_CB[i][j])
                print 'e'

    #aplicacao da tecnica de mapeamento
    def technique_mapping(self):
        M = self.N / self.NB
        # array = np.identity(M, dtype=np.int)
        # array = np.eye(M, M, k=self.matrix_basic, dtype=int)
        array_CB =  np.vsplit(self.matrix_basic, self.NB+1)
        array_Z = np.vsplit(self.matrix_zeros,self.NB+1)

    def transform_m_basic(self):
        arr = []
        for i in self.matrix_basic:
            arr.append(list(i))
        return arr

    def transform_m_zero(self):
        arr = []
        for i in self.matrix_zeros:
            arr.append(list(i))
        return arr

    @staticmethod
    def int_row(row):
        return map(lambda x: int(x), row)

    def test_method(self, valueM):
        arr = []
        for position in xrange(valueM):
            for i in range(self.NB):
                arr2 = []
                for j in xrange(valueM):
                    if j == position:
                        arr2 += Code_class.int_row(self.matrix_basic[i])
                    else:
                        arr2 += list(self.matrix_zeros[i])
                arr.append(arr2)
        return arr


    def capeta_tecnica_de_mapeamento(self):
        M = self.N / self.NB
        array_of_doom = []
        array_final = [[1, 0], [0, 1]]

        i = 0
        j = 0
        array_of_doom.append(list(self.matrix_basic[0]) + list(self.matrix_zeros[0]))
        arr = self.test_method(M)
        #for row in arr:
        #    print row
        #todos os elementos da coluna
        #print [y[0] for y in arr]

        return arr
    def verify_same_sector(self,Ci,Cj,array_codes):
        count_NB = 0

        while count_NB <= len(array_codes):

            if Ci != Cj:
                if(Ci > count_NB and Ci <= count_NB+ self.NB ):
                    if (Cj > count_NB and Cj <= count_NB + self.NB):
                        return True
                count_NB = count_NB+ self.NB
        return False




