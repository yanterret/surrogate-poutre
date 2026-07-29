import numpy as np

from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler

import pandas as pd

NOMS = ['L', 'b', 'h', 'P', 'E']
bornes_inf = np.array([1000.0, 20.0, 20.0])
bornes_supp = np.array([2000.0, 100.0, 100.0])

# on genere les points que l on va simuler
def generer_doe(n, seed):
    PE = np.hstack((np.full(n,2000).reshape(-1, 1),np.full(n,210000).reshape(-1, 1)))
    X_inter = np.empty((0, 3))
    nb = n
    seed_bis = seed  # on change la seed car si dans la boucle on demande le meme nombre de points ça renverra la meme chose donc si ne marchent pas ne changerontr rien 
    while np.shape(X_inter)[0] < n :
        X_append =  qmc.scale(qmc.LatinHypercube(d=3, seed=seed_bis).random(nb), LO, HI)
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

# attention aux seeds
X_train = generer_doe(30, 0)  # seeds pairs pour éviter la superposition avec les seed test
X_test  = generer_doe(20, 1) # seed impaires

def exporter_doe(X, chemin):
    with open(chemin, mode='w', encoding='utf8') as F:
        F.write('id;L;b;h;P;E' + '\n')
        for i, x in enumerate(X):
            ligne = f"{i:03d}" # pour respecter le format abaqus...
            for v in x:
                ligne += ';' + f"{v:.4f}" # de meme
            F.write(ligne + '\n')

exporter_doe(X_train, 'doe_train3.csv')
exporter_doe(X_test, 'doe_test3.csv')   # données avant delta et sigma max donné par abaqus

# à la fin des simulations on retransforme en array 
def csv_to_array(doc):
    
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

# création des données de testà partir de simulations abaqus

Ytest, Xtest  = csv_to_array('doe_test3_complet.csv')
Ytr, Xtr = csv_to_array('doe_train30_bon.csv')

sx = StandardScaler().fit(Xtr) # on fit ( moyenne écart type) sur le train car il y a des hypotheses dans le modéle sur la moyenne de Y 
Xtr_n, Xtest_n = sx.transform(Xtr), sx.transform(Xtest) # création d un set normalisé meilleur pour le fit polynomiale
