
#################
# Libraries
#################
import numpy as np
from smt.surrogate_models import KRG
 
# code surrogate de base
 
#################
# paramètres de la simulation
#################
DELTA_MAX = 5.0      # mm on le pose pour satisfaire la rdm
SIGMA_MAX = 200.0    # MPa donné par la rdm
L0 = 1500.0          # mm entre 1000 et 2000 ok
RHO = 7850e-9        # kg/mm3 acier
 
#################
# surrogate kriging avec un coefficient de marge 0.95
#################
def surrogate_kriging(y, gp, sx):   # gp et sx passés en argument via args= de differential_evolution
    '''
    entrée :
    Xtr_n,Ytr,Ytest,Xtest_n : array
 
    sortie :
    Yp : array, Ytest sert uniquement à donner la forme
    '''
    b, h = y
    d, s = np.exp(gp.predict(sx.transform([[L0, b, h]]))[0])   # fleche, contrainte
    if d <= DELTA_MAX *0.95 and s <= SIGMA_MAX:
        return RHO * b * h * L0      # poutre valide
    return 1e10 # on mets une valeure absudément haute pour que l optimizer la considére comme mauvaise
 
#################
# en utilisant la variance donnée par le krigiage 
#################
def surrogate_kriging_variance(y, gp, sx):   # renommé : sinon il écrase la fonction du dessus
    '''
    entrée :
    Xtr_n,Ytr,Ytest,Xtest_n : array
 
    sortie :
    Yp : array, Ytest sert uniquement à donner la forme
    '''
    b, h = y
    valeurs, std = gp.predict(sx.transform([[L0, b, h]]), return_std=True) # valeurs = d, s
    delta = np.exp(valeurs[0, 0] + std[0, 0])       # valeur de la fleche pénalisé par la variance
    sigma = np.exp(valeurs[0, 1] + std[0, 1])       # valeur du sigma max de meme
    if delta <= DELTA_MAX and sigma <= SIGMA_MAX:   # plus besoin du 0.95 : la marge vient de la variance
        return RHO * b * h * L0      # poutre valide
    return 1e10
 
#################
# kriging smt avec traning une sortie par une sortie sinon le résulats n est aps satisfaisant
#################
def entrainer(sortie, Xtr_n, Ytr):
    '''
    entrée :
    Xtr_n,Ytr,Ytest,Xtest_n : array
 
    sortie :
    Yp : array, Ytest sert uniquement à donner la forme
    '''
    smt = KRG(print_global=False)
    smt.set_training_values(Xtr_n, np.log(Ytr[:, sortie]))
    smt.train()
    return smt
 
def surrogate_kriging_smt(y, gp_delta, gp_sigma, sx):
    '''
    entrée :
    Xtr_n,Ytr,Ytest,Xtest_n : array
 
    sortie :
    Yp : array, Ytest sert uniquement à donner la forme
    '''
    b, h = y
    x = sx.transform([[L0, b, h]])
    d, s = np.exp(gp_delta.predict_values(x)), np.exp(gp_sigma.predict_values(x))
    if d <= DELTA_MAX and s <= SIGMA_MAX:
        return RHO * b * h * L0      # poutre valide
    return 1e9
 
#################
# code pour vérifier avec la rdm 
#################
def rdm(L, b, h, P, E):
    '''
    entrée :
    Xtr_n,Ytr,Ytest,Xtest_n : array
 
    sortie :
    Yp : array, Ytest sert uniquement à donner la forme
    '''
    I =b * h**3 / 12.0
    delta = P*L**3 /(3*E*I)
    sigma = P * L * (h / 2.0) / I
    masse = RHO * b * h * L
    return delta, sigma, masse