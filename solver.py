from traceback import print_tb
import funcoesTermosol as ft
import numpy as np
from math import *

class Elemento():
    def __init__(self, numero, L, co, si, E, A, nos,n1, n2, ke):
        self.numero = numero
        self.L = L
        self.co = co
        self.si = si
        self.E = E
        self.A = A
        self.nos = nos
        self.n1 = n1
        self.n2 = n2
        self.ke = ke

    #Função que calcula matriz de rigidez do elemento  
    def calculaKe(self):
        c = self.co
        s = self.si

        K = [[c**2, c*s, -1*(c**2), -1*c*s],
             [c*s, s**2, -1*c*s, -s**2],
             [-1*(c**2), -1*c*s, c**2, c*s],
             [-1*s*c, -1*(s**2), c*s, s**2]]

        K = np.round(K, 5)

        kele = np.multiply(K, ((self.A*self.E)/self.L) )
        self.ke = kele 

#Função que calcula matriz de liberdade
def Jacobi_geral(ite, tol, K, F):
    lista_erros = []
    linhas, colunas = np.shape(K)
    U = np.zeros((linhas,1))
    U_novo = np.zeros((linhas,1))
    contador = 0
    while contador < ite:
        for i in range(linhas):
            U_novo[i] = F[i]
            for j in range(colunas):
                if i != j:
                    U_novo[i] -= K[i][j] * U[j]

            if K[i][i] != 0:
                U_novo[i] /= K[i][i]

            if U_novo[i] != 0:
                lista_erros.append((U[i] - U_novo[i])/U_novo[i])

            if U_novo[i] != 0:
                if abs((U[i] - U_novo[i])/U_novo[i]) < tol:
                    ei = np.max(lista_erros)
                    return U, ei     
        
        for i in range(linhas):
            U[i] = U_novo[i]
        contador += 1
    ei = np.max(lista_erros)
    return U, ei

def Gauss_geral(ite, tol, K, F):
    contador = 0
    linhas, colunas = np.shape(K)
    U = np.zeros((linhas,1))
    U_novo = np.zeros((linhas,1))
    lista_erros = np.ones((linhas,1))
    while contador < ite:
        for i in range(linhas):
           U[i] = U_novo[i]
        for i in range(linhas):
            soma = 0
            for j in range(colunas):
                if j != i:
                    soma += K[i][j] * U_novo[j]
                
            U_novo[i] = (F[i] - soma)/K[i][i]

            if U_novo[i] != 0:
                lista_erros[i] = abs((U_novo[i] - U[i])/U_novo[i])
        erro = np.amax(lista_erros)
        contador+=1
        if erro <= tol:
            break
    
    return U_novo, erro

[nn, N, nm, Inc, nc, F, nr, R] = ft.importa('entrada.xls')

Pnos = list(map(tuple, N.T))

elementos = []
zeros = np.zeros((4, 4))
ite = 1e3
tol = 1e-8

i = 0
for k in Inc:
    n1 = Pnos[int(k[0]-1)]
    n2 = Pnos[int(k[1]-1)]
    L = dist(Pnos[int(k[0]-1)] ,  Pnos[int(k[1]-1)])
    co = (Pnos[int(k[1]-1)][0] - Pnos[int(k[0]-1)][0]) / L
    si = (Pnos[int(k[1]-1)][1] - Pnos[int(k[0]-1)][1]) / L
    E = k[2]
    A = k[3]
    nos = (k[0], k[1])
    elementos.append(Elemento((i + 1), L, co, si, E, A, nos, n1, n2, zeros))
    i += 1

for ele in elementos:
    ele.calculaKe()

Kg = np.zeros((nn*2, nn*2))

GDL = []
for ele in elementos:
    #Monta matriz GDL
    GDL.append([elementos[int(ele.nos[0]) - 1].numero*2 - 2,
                elementos[int(ele.nos[0]) - 1].numero*2 - 1,
                elementos[int(ele.nos[1]) - 1].numero*2 - 2,
                elementos[int(ele.nos[1]) - 1].numero*2 - 1])


conta1 = 0
conta2 = 0
for ele in elementos:
    conta1 = 0
    for i in GDL[ele.numero-1]:
        conta2 = 0
        for j in GDL[ele.numero-1]:
            #Monta matriz global de rigidez
            Kg[i][j] += ele.ke[conta1][conta2] 
            conta2 += 1
        conta1 += 1


R = R.astype(int)

#Guarda matriz de rigidez global
Kg_original = Kg.copy()

#Aplicando restrições à matriz de rigidez
Kg = np.delete(Kg, R, axis=1)
Kg = np.delete(Kg, R, axis=0)

U_novo = F.copy()

#Aplicando restrições à matriz de forças
F = np.delete(F, R, axis=0)

#Conseguindo matriz de deslocamentos U, após aplicação das restrições
U, erro_maximo = Gauss_geral(ite, tol, Kg, F)

conta3 = 0
lista_incluir = []
while conta3 < len(U_novo):
    if conta3 not in R:
        lista_incluir.append(conta3)
    conta3+=1


for i in R:
    U_novo[i[0]] = 0

conta4 = 0
for i in lista_incluir:
    #Adiciona valores à matriz de deslocamento global
    U_novo[i] = U[conta4]
    conta4+=1


#Calculando reações de apoio
reacoes = np.matmul(Kg_original, U_novo)
reacoes = np.delete(reacoes, lista_incluir, axis=0)
reacoes = np.round(reacoes)

deformacao = np.empty(len(elementos))

tensao = np.empty(len(elementos))

internas = np.empty(len(elementos))

for ele in elementos:
    #Calculando deformações
    deformacao[ele.numero-1] = (np.matmul([-ele.co, -ele.si, ele.co, ele.si],
    [U_novo[GDL[ele.numero-1][0]], U_novo[GDL[ele.numero-1][1]], U_novo[GDL[ele.numero-1][2]], U_novo[GDL[ele.numero-1][3]]]) / ele.L)

    #Calculando tensões internas
    tensao[ele.numero-1] = (ele.E * np.matmul([-ele.co, -ele.si, ele.co, ele.si],
    [U_novo[GDL[ele.numero-1][0]], U_novo[GDL[ele.numero-1][1]], U_novo[GDL[ele.numero-1][2]], U_novo[GDL[ele.numero-1][3]]]) / ele.L)

    # Calculando forças internas
    internas[ele.numero-1] = ele.A * tensao[ele.numero-1]

deformacao = np.split(deformacao, len(deformacao))
deformacao = np.stack(deformacao)

tensao = np.split(tensao, len(deformacao))
tensao = np.stack(tensao)

internas = np.split(internas, len(deformacao))
internas = np.stack(internas)


print("Reações de apoio [N]")
print(reacoes)

print("Deslocamentos [m]")
print(U_novo)

print("Deformações []")
print(deformacao)

print("Forças internas [N]")
print(internas)

print("Tensões internas [Pa]")
print(tensao)



ft.plota(N, Inc)

ft.geraSaida("APS4_Grupo7", reacoes, U_novo, deformacao, internas, tensao)
#ft.geraSaida(nome, Ft, Ut, Epsi, Fi, Ti)