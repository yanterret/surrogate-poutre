import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.optimize import differential_evolution

from generation_des_entrees import generer_doe, exporter_doe, csv_to_array
from methodes_resolutions import SORTIES, evaluer
from surrogate import (surrogate_kriging, surrogate_kriging_variance,
                       entrainer, surrogate_kriging_smt, rdm,
                       DELTA_MAX, SIGMA_MAX, L0, RHO)

# chemins bases sur l emplacement de ce fichier : marche depuis n importe ou
RACINE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(RACINE, 'data')

RESULTATS = []   # (methode, sortie, R2, ERM) -> ecrit en csv a la fin


# ---------- generation des entrees ----------

# attention aux seeds
X_train = generer_doe(30, 0)  # seeds pairs pour éviter la superposition avec les seed test
X_test  = generer_doe(20, 1) # seed impaires

exporter_doe(X_train, os.path.join(DATA, 'doe_train3.csv'))
exporter_doe(X_test, os.path.join(DATA, 'doe_test3.csv'))   # données avant delta et sigma max donné par abaqus

# création des données de test à partir de simulations abaqus

Ytest, Xtest  = csv_to_array(os.path.join(DATA, 'doe_test3_complet.csv'))
Ytr, Xtr = csv_to_array(os.path.join(DATA, 'doe_train30_bon.csv'))

sx = StandardScaler().fit(Xtr) # on fit ( moyenne écart type) sur le train car il y a des hypotheses dans le modéle sur la moyenne de Y
Xtr_n, Xtest_n = sx.transform(Xtr), sx.transform(Xtest) # création d un set normalisé meilleur pour le fit polynomiale


# ---------- résolution par polynomes ----------

for deg in [1, 2, 3, 4,5,6]:
    poly = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    poly.fit(Xtr_n, Ytr)
    res = evaluer(Ytest, poly.predict(Xtest_n), f'poly{deg}')
    RESULTATS += [(f'poly{deg}', s, *res[s]) for s in SORTIES]

poly_log = make_pipeline(PolynomialFeatures(1), LinearRegression())
poly_log.fit(np.log(Xtr), np.log(Ytr))
Y_pred_log = np.exp(poly_log.predict(np.log(Xtest)))

res = evaluer(Ytest, Y_pred_log, 'poly_log')
RESULTATS += [('poly_log', s, *res[s]) for s in SORTIES]


# ---------- kriging ----------

kernel = C(1) * RBF(length_scale=np.ones(3)) + WhiteKernel(1e-4, (1e-6, 1e-2))
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gp.fit(Xtr_n, np.log(Ytr))

Y_pred_gp = np.exp(gp.predict(Xtest_n)) # array (20, 2)

res = evaluer(Ytest, Y_pred_gp, 'kriging')
RESULTATS += [('kriging', s, *res[s]) for s in SORTIES]


# ---------- smt ----------

from smt.surrogate_models import KRG

Yp = np.zeros_like(Ytest)

for j, s in enumerate(SORTIES):
    sm = KRG(print_global=False,theta0=[1e-2],theta_bounds = [1e-4,1e0])
    sm.set_training_values(Xtr_n, np.log(Ytr[:, j]))
    sm.train()
    y =np.exp( sm.predict_values(Xtest_n)) # array (20, 1)
    Yp[:, j] = y.ravel()

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

with open(os.path.join(DATA, 'resultats.csv'), mode='w', encoding='utf8') as F:
    F.write('methode;sortie;R2;ERM\n')
    for methode, sortie, r_2, err in RESULTATS:
        F.write(f'{methode};{sortie};{r_2:.4f};{err:.2f}\n')