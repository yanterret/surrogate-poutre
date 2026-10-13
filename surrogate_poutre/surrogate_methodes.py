#################
# Libraries
#################
import numpy as np
from smt.surrogate_models import KRG

#################
# paramètres
#################
delta_max = 5.0      # mm on le pose pour satisfaire la rdm
sigma_max= 200.0    # MPa donné par la rdm
l0 = 1500.0          # mm entre 1000 et 2000 ok
rho = 7850e-9        # kg/mm3 acier

#################
# surrogate kriging avec un coefficient de marge 0.95
#################
def surrogate_kriging(y, gp, sx,DELTA_MAX=delta_max,SIGMA_MAX = sigma_max,L0 = l0,RHO = rho):   # gp et sx passés en argument via args= de differential_evolution
    '''
    entrée :
    y : array shape(2,) : (b, h)
    gp : GaussianProcessRegressor entraîné sur log(Y)
    sx : StandardScaler entraîné sur les données

    sortie :
    masse : float : kg
    '''
    b, h = y
    d, s = np.exp(gp.predict(sx.transform([[L0, b, h]]))[0])   # flèche, contrainte
    if d <= DELTA_MAX *0.95 and s <= SIGMA_MAX:
        return RHO * b * h * L0      # poutre valide
    return 1e10 # on met une valeur absurdement haute pour que l'optimizer la considère comme mauvaise

#################
# en utilisant la variance donnée par le krigeage
#################
def surrogate_kriging_variance(y, gp, sx,DELTA_MAX=delta_max,SIGMA_MAX = sigma_max,L0 = l0,RHO = rho):   # renommé : sinon il écrase la fonction du dessus
    '''
    entrée :
    y : array shape(2,) : (b, h)
    gp : GaussianProcessRegressor entraîné sur log(Y)
    sx : StandardScaler entraîné sur les données

    sortie :
    masse : float : kg
    '''
    b, h = y
    valeurs, std = gp.predict(sx.transform([[L0, b, h]]), return_std=True) # valeurs = d, s
    delta = np.exp(valeurs[0, 0] + std[0, 0])       # valeur de la flèche pénalisée par la variance
    sigma = np.exp(valeurs[0, 1] + std[0, 1])       # valeur du sigma max de même
    if delta <= DELTA_MAX and sigma <= SIGMA_MAX:   # plus besoin du 0.95 : la marge vient de la variance
        return RHO * b * h * L0      # poutre valide
    return 1e10

#################
# kriging smt avec entraînement une sortie par une sortie sinon le résultat n'est pas satisfaisant
#################
def entrainer(sortie, Xtr_n, Ytr):
    '''
    entrée :
    sortie : int : 0 pour delta, 1 pour sigma
    Xtr_n : array shape(n,3)
    Ytr : array shape(n,2)

    sortie :
    smt : modèle entraîné
    '''
    smt = KRG(print_global=False)
    smt.set_training_values(Xtr_n, np.log(Ytr[:, sortie]))
    smt.train()
    return smt

def surrogate_kriging_smt(y, gp_delta, gp_sigma, sx,DELTA_MAX=delta_max,SIGMA_MAX = sigma_max,L0 = l0,RHO = rho):
    '''
    entrée :
    y : array shape(2,) : (b, h)
    gp_delta, gp_sigma : modèles smt entraînés sur log(Y), un par sortie
    sx : StandardScaler entraîné sur les données

    sortie :
    masse : float : kg
    '''
    b, h = y
    x = sx.transform([[L0, b, h]])
    d, s = np.exp(gp_delta.predict_values(x)), np.exp(gp_sigma.predict_values(x))
    if d <= DELTA_MAX and s <= SIGMA_MAX:
        return RHO * b * h * L0      # poutre valide
    return 1e10

#################
# code pour vérifier avec la rdm
#################
def rdm(L, b, h, F, E,RHO = rho):
    '''
    entrée :
    L, b, h, F, E, RHO : float

    sortie :
    delta, sigma, masse : float :  en mm,  en MPa, en kg
    '''
    I =b * h**3 / 12.0
    delta = F*L**3 /(3*E*I)
    sigma = F * L * (h / 2.0) / I
    masse = RHO * b * h * L
    return delta, sigma, masse