#################
# Libraries
#################

import numpy as np
from scipy.stats import qmc
import pandas as pd

#################
# paramètres de la simulation
#################
bornes_inf = np.array([1000.0, 20.0, 20.0])
bornes_supp = np.array([2000.0, 100.0, 100.0])

#################
# génération des points de simulations
#################
def generer_doe(n, seed):
    '''
    entrée : 
    n : int : nombre de points
    seed : int : numéro de la seed

    sortie :
    X : array shape(n,5) dedans des float 
    '''
    PE = np.hstack((np.full(n,2000).reshape(-1, 1),np.full(n,210000).reshape(-1, 1)))
    X_inter = np.empty((0, 3))
    nb = n
    seed_bis = seed  # on change la seed car si dans la boucle on demande le meme nombre de points ça renverra la meme chose donc si ne marchent pas ne changerontr rien 
    while np.shape(X_inter)[0] < n :
        X_append =  qmc.scale(qmc.LatinHypercube(d=3, seed=seed_bis).random(nb), bornes_inf, bornes_supp)
        L, b, h = X_append[:,0], X_append[:,1],X_append[:,2]
        P = 2000
        E = 210000
        I = b * h**3 / 12
        delta = P * L**3 / (3 * E * I)
        sigma = P * L * (h/2) / I
        elancement = L / h
        X_bon = X_append[(elancement >= 10)&(delta/L <= 0.05)&(sigma <= 200)] 
        X_inter = np.vstack((X_inter , X_bon))
        nb = max(n - np.shape(X_inter)[0],1)   # max car peut etre négatif..
        seed_bis += 2
    return np.hstack((X_inter,PE))[:n]  # shape(n,5)

def exporter_doe(X, chemin):
    '''
    entrée : 
    X : array shape(n,5) 
    chemin : string

    sortie :
    fichier csv via chemin
    '''
    with open(chemin, mode='w', encoding='utf8') as F:
        F.write('id;L;b;h;P;E' + '\n')
        for i, x in enumerate(X):
            ligne = f"{i:03d}" # pour respecter le format abaqus...
            for v in x:
                ligne += ';' + f"{v:.4f}" # de meme
            F.write(ligne + '\n')

# à la fin des simulations on retransforme en array 
def csv_to_array(doc):
    '''
    entrée : 
    doc : csv

    sortie :
    Y : array shape(n,3) dedans des float 
    X : array shape(n,3) dedans des float 
    '''    
    df = pd.read_csv(doc, sep=';', encoding='utf-8')
    df.columns = df.columns.str.strip()
    L = df['L'].to_numpy().reshape(-1, 1)
    b = df['b'].to_numpy().reshape(-1, 1)
    h = df['h'].to_numpy().reshape(-1, 1)
    
    Delta = df['Delta'].to_numpy().reshape(-1, 1)
    Sigma_max = df['Sigma_max'].to_numpy().reshape(-1, 1)
    
    Y = np.hstack([Delta, Sigma_max])
    X = np.hstack([L, b, h])  
    
    return Y,X

#################
# métriques
#################
SORTIES = ['delta', 'sigma']

def r2(y_vrai, y_pred):
    '''
    entrée : 
    y_vrai, y_pred : float ou array 

    sortie :
    R2 : float
    '''
    ss_res = np.sum((y_vrai - y_pred)**2)
    ss_tot = np.sum((y_vrai - y_vrai.mean())**2)
    return 1 - ss_res/ss_tot

def evaluer(Y_vrai, Y_pred):
    '''
    entrée : 
    y_vrai, y_pred : float ou array 

    sortie :
    métriques : string final
    '''
    res = {}
    for j, s in enumerate(SORTIES):
        r_2 = r2(Y_vrai[:, j], Y_pred[:, j])
        err = np.median(np.abs(Y_pred[:, j] - Y_vrai[:, j]) / np.abs(Y_vrai[:, j])) * 100
        res[s] = (r_2, err)
        print(f'{s} R² = {r_2}, ERM = {err} %')
    return res
