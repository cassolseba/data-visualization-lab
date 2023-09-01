#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# lab02_ml.py
#
# Libreria contenente alcuni algoritmi di machine learning
#
# Esecuzione:
#    Importare la libreria nel programma principale con "import ml"
#    Vedere lab02_iris2.py come esempio d'uso.
#
# Attenzione: il codice serve esclusivamente a scopi didattici
# e non è adatto a un utilizzo "serio": mancano molti accorgimenti e ottimizzazioni
# che lo renderebbero molto più complesso e difficile da capire.

############################################
#
# Moduli

# Conteggio delle occorrenze di un elemento in un vettore (utile per KNN)
from collections import Counter

############################################
#
# Funzioni di utilità

# Distanza euclidea al quadrato fra i due vettori x1 e x2
def distance2 (x1, x2):
	return sum((a-b)**2 for a,b in zip(x1,x2))

############################################
#
# K-Nearest Neighbors
#
# Definiamo la classe KNN contenente un costruttore __init__, una funzione di addestramento fit
# e una funzione di valutazione di un nuovo vettore predict.

class KNN:

	# Costruttore: si limita a memorizzare il parametro K nell'oggetto
	def __init__ (self, K):
		self.K = K

	# Addestramento: si limita a memorizzare il dataset X,y nell'oggetto
	def fit (self, X, y):
		self.X = X
		self.y = y

	# Valutazione del modello (predizione): dato il vettore incognito x1,
	# ne stima la classe trovando quella più rappresentata fra i K elementi
	# del dataset di addestramento più vicini a x1
	def predict (self, x1):
		# Crea la lista delle coppie (d_i,y_i) dove i itera sul dataset, d_i è
		# la distanza dell'i-esimo elemento del dataset da x1, e y_i è la sua classe. 
		d = [
			(distance2(v,x1),c)
			for v,c in zip(self.X.as_matrix(),self.y)
		]
		# Ordina la lista per distanza crescente e ne estrae i primi K elementi.
		sorted_d = sorted(d)[:self.K]
		# Recupera le sole informaazioni di classe, scartando le distanze.
		classes = [c for d,c in sorted_d]
		# Conta le occorrenze di ciascuna classe
		count = Counter(classes)
		# Restituisce la classe più rappresentata.
		return count.most_common(1)[0][0]

##################################################
#
# Minimi quadrati a una dimensione
#
# Definiamo la classe LSQ1D contenente una funzione di addestramento fit
# e una funzione di valutazione di un nuovo vettore predict.
# Non è necessario definire il costruttore perché non ci sono parametri da impostare.

class LSQ1D:

	def fit (self, x, y):
		xm = x.as_matrix()
		ym = y.as_matrix()
		self.beta = sum(a*b for a,b in zip(xm,ym)) / sum(a*a for a in xm)

	def predict (self, x):
		return self.beta * x
