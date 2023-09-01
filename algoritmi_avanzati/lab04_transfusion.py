#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# lab04_transfusion.py
#
# Esperimenti con la regressione logistica, discesa lungo il gradiente
#
# Richiede la presenza del file "transfusion.data" (dal repository UCI)
# nella stessa directory
#
# Esecuzione:
#    python2 lab04_transfusion.py
# oppure
#    chmod 755 lab04_transfusion.py   <--- una tantum
#    ./lab04_transfusion.py
#
# Attenzione: il codice serve esclusivamente a scopi didattici
# e non è adatto a un utilizzo "serio": mancano molti accorgimenti e ottimizzazioni
# che lo renderebbero molto più complesso e difficile da capire.

############################################
#
# Moduli

# Lettura del file CSV
import pandas

# Algebra delle matrici
import numpy as np

############################################
#
# Lettura dei dati

# Il file è già in formato CSV corretto, con una riga di intestazione,
# ma i dati sono interi, quindi dobbiamo specificare che
# li vogliamo in virgola mobile.
dataset = pandas.read_csv('transfusion.data', dtype=np.float64)

# Estraiamo le prime 4 colonne come X, l'ultima (0 e 1) come y
X = dataset.iloc[:,:4].as_matrix()
y = dataset.iloc[:,4].as_matrix()

# Normalizziamo le colonne di input riscalandole rispetto al loro massimo
for i in range(4):
    X[:,i] /= np.max(X[:,i])


############################################
#
# Funzioni

# La funzione sigmoide 1 / (1 + e^-t)
def sigmoide(t):
    return 1.0 / (1.0 + np.exp(-t))

# Data una matrice di input e un vettore di pesi,
# calcola il vettore delle previsioni del modello logit
def previsione(X, beta):
    return sigmoide(X.dot(beta))

# Dato il vettore degli output desiderati y, quello
# degli output previsti yp, la matrice di input e
# l'indice del peso k, calcola la derivata parziale rispetto a beta_k
# della soma degli scarti al quadrato
def derivata_parziale(y, yp, X, k):
    return 2.0 * np.sum((yp - y) * yp * (1.0 - yp) * X[:,k])

# Costruisce il vettore delle derivate parziali
def gradiente(y, yp, X):
    return np.array([derivata_parziale(y, yp, X, k)
                        for k in range(4)])

# Dato il vettore degli output desiderati y e quello
# degli output previsti yp, restituisce l'RMSE.
def RMSE(y, yp):
    return np.sqrt(np.sum((y - yp)**2) / len(y))

############################################
#
# Discesa lungo il gradiente

# Inizializzazione dei pesi: beta può partire da valori casuali, oppure da zero.
#beta = np.zeros(4)
beta = np.random.random(4)

# Learning rate iniziale
eta = 1.0e-10

# Prima previsione e RMSE corrispondente
yp = previsione(X,beta)
errore = RMSE(y, yp)

# Ripetizione del passo di discesa
while True:

    # Calcolo del gradiente
    grad = gradiente(y, yp, X)

    # Ricerca di un passo eta adeguato
    while True:

        # Sposta il vettore dei pesi contro il gradiente
        beta1 = beta - eta * grad

        # Calcola la nuova previsione e il nuovo errore
        yp1 = previsione(X, beta1)
        errore1 = RMSE(y, yp1)

        # Se il passo peggiora le cose, allora eta è troppo grande: dimezzare e riprovare
        if errore1 >= errore:
            eta /= 2

        # Altrimenti, eta è adeguato
        else:
            # Aumentare leggermente eta per la prossima iterazione
            eta *= 1.1
            # Aggiornare i valori di beta, yp ed errore
            errore = errore1
            beta = beta1
            yp = yp1
            # Uscire dal ciclo interno e passare all'iterazione successiva
            break

    # Stampa l'errore corrente prima di ripetere il passo
    print (errore, eta)
