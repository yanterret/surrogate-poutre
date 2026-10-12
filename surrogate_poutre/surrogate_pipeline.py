#################
# Libraries
#################
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.optimize import differential_evolution
from smt.surrogate_models import KRG

#################
# importation des fonctions
#################
from generation_des_entrees import generer_doe, exporter_doe, csv_to_array
from methodes_resolutions import evaluer
from surrogate import surrogate_kriging, surrogate_kriging_variance,entrainer, surrogate_kriging_smt, rdm
                       
#################
# paramètres de la simulation
#################
DELTA_MAX = 5.0      # mm on le pose pour satisfaire la rdm
SIGMA_MAX = 200.0    # MPa donné par la rdm
L0 = 1500.0          # mm entre 1000 et 2000 ok
RHO = 7850e-9        # kg/mm3 acier
SORTIES = ['delta', 'sigma']

RESULTATS = []   # (methode, sortie, R2, ERM) -> ecrit en csv a la fin

#################
# generation des données d'entrees
#################
X_train = generer_doe(30, 0)  # seeds pairs pour éviter la superposition avec les seed test
X_test  = generer_doe(20, 1) # seed impaires
exporter_doe(X_train, "data/doe_train.csv")
exporter_doe(X_test, "data/doe_test.csv")   # données avant delta et sigma max donné par abaqus

# création des données de test à partir de simulations abaqus
Ytest, Xtest  = csv_to_array("data/doe_test3_complet.csv")
Ytr, Xtr = csv_to_array("data/doe_train30_bon.csv")
sx = StandardScaler().fit(Xtr) # on fit ( moyenne écart type) sur le train car il y a des hypotheses dans le modéle sur la moyenne de Y
Xtr_n, Xtest_n = sx.transform(Xtr), sx.transform(Xtest) # création d un set normalisé meilleur pour le fit polynomiale

#################
# méthodes de résolutions
#################
# par polynomes et log polynomes
resol_polynomes(Xtr_n,Ytr,Ytest,Xtest_n)
 
res = evaluer(Ytest, poly.predict(Xtest_n), f'poly{deg}')
RESULTATS += [(f'poly{deg}', s, *res[s]) for s in SORTIES]
res = evaluer(Ytest, Y_pred_log, 'poly_log')
RESULTATS += [('poly_log', s, *res[s]) for s in SORTIES]

# par krigegage
resol_kriging(Xtr_n,Ytr,Ytest,Xtest_n):

res = evaluer(Ytest, Y_pred_gp, 'kriging')
RESULTATS += [('kriging', s, *res[s]) for s in SORTIES]


# par krigegage smt
resol_kriging_smt(Xtr_n,Ytr,Ytest,Xtest_n):

res = evaluer(Ytest, Yp, 'smt')
RESULTATS += [('smt', s, *res[s]) for s in SORTIES]


# ---------- utilisation : optimisation de la section ----------

# args= passe gp et sx a la fonction objectif, que differential_evolution
# n appelle qu avec le vecteur (b, h)

res_opt = differential_evolution(surrogate_kriging, [(20, 100), (20, 100)], args=(gp, sx)) # on test les poutre
b, h = res_opt.x
delta, sigma, masse = rdm(L0, b, h, 2000, 210000)
print(f'kriging coef 0.95   b = {b:.1f} mm, h = {h:.1f} mm, masse = {masse:.2f} kg, delta reel = {delta:.3f} mm')

res_opt = differential_evolution(surrogate_kriging_variance, [(20, 100), (20, 100)], args=(gp, sx))
b, h = res_opt.x # sort les valeurs de differential_evol...
delta, sigma, masse = rdm(L0, b, h, 2000, 210000)
print(f'kriging variance    b = {b:.1f} mm, h = {h:.1f} mm, masse = {masse:.2f} kg, delta reel = {delta:.3f} mm')

gp_delta, gp_sigma = entrainer(0, Xtr_n, Ytr), entrainer(1, Xtr_n, Ytr)

res_opt = differential_evolution(surrogate_kriging_smt, [(20, 100), (20, 100)], args=(gp_delta, gp_sigma, sx))
b, h = res_opt.x # sort les valeurs de differential_evol...
delta, sigma, masse = rdm(L0, b, h, 2000, 210000)
print(f'smt sans marge      b = {b:.1f} mm, h = {h:.1f} mm, masse = {masse:.2f} kg, delta reel = {delta:.3f} mm')


# ---------- ecriture des resultats en csv ----------

with open("resultats/florileche_resultat.csv", mode='w', encoding='utf8') as F:
    F.write('methode;sortie;R2;ERM\n')
    for methode, sortie, r_2, err in RESULTATS:
        F.write(f'{methode};{sortie};{r_2:.4f};{err:.2f}\n')