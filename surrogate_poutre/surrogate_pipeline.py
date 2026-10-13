#################
# Libraries
#################
import os
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution
import numpy as np
#################
# importation des fonctions
#################
from generation_des_entrees import generer_doe, exporter_doe, csv_to_array, evaluer
from methodes_resolutions import resol_polynomes, resol_kriging, resol_kriging_smt
from surrogate_methodes import surrogate_kriging, surrogate_kriging_variance, entrainer, surrogate_kriging_smt, rdm

#################
# paramétres de la simulation
#################
# dans methodes_resolutions
SORTIES = ['delta', 'sigma']
L0 = 1500.0 
RESULTATS = []   # (methode, sortie, R2, ERM)

# dans generation_des_entrees
bornes_inf = np.array([1000.0, 20.0, 20.0])
bornes_supp = np.array([2000.0, 100.0, 100.0])
E = 210000 # MPA
F = 2000 #  N
# dans surrogate_methodes
delta_max = 5.0      # mm donné par la rdm
sigma_max= 200.0    # MPa donné par la rdm
l0 = 1500.0          # mm entre 1000 et 2000 ok
borne_opti_b , borne_opti_h = (bornes_inf[1],bornes_supp[1]) , (bornes_inf[2],bornes_supp[2])

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
resu_poly_deg, Y_pred_log = resol_polynomes(Xtr, Ytr, Xtest, Xtr_n, Xtest_n)

for deg, Ypred_poly in enumerate(resu_poly_deg, start=1):   # start=1 : la liste commence au degré 1
    res = evaluer(Ytest, Ypred_poly, f'poly{deg}')
    RESULTATS += [(f'poly{deg}', s, *res[s]) for s in SORTIES]

res = evaluer(Ytest, Y_pred_log, 'poly_log')
RESULTATS += [('poly_log', s, *res[s]) for s in SORTIES]

# par krigegage
Y_pred_gp, gp = resol_kriging(Xtr_n, Ytr, Xtest_n)   # gp resert pour l optimisation
res = evaluer(Ytest, Y_pred_gp, 'kriging')
RESULTATS += [('kriging', s, *res[s]) for s in SORTIES]

# par krigegage smt
Yp = resol_kriging_smt(Xtr_n, Ytr, Ytest, Xtest_n)
res = evaluer(Ytest, Yp, 'smt')
RESULTATS += [('smt', s, *res[s]) for s in SORTIES]

#################
# utilisation : optimisation de la section
#################
# args= passe gp et sx a la fonction objectif, que differential_evolution
# n appelle qu avec le vecteur (b, h)

res_opt = differential_evolution(surrogate_kriging, [borne_opti_b, borne_opti_h], args=(gp, sx)) # on test les poutre
b, h = res_opt.x
delta, sigma, masse = rdm(L0, b, h, F, E)
print(f'kriging coef 0.95   b = {b:.1f} mm, h = {h:.1f} mm, masse = {masse:.2f} kg, delta reel = {delta:.3f} mm')

res_opt = differential_evolution(surrogate_kriging_variance, [borne_opti_b, borne_opti_h], args=(gp, sx))
b, h = res_opt.x # sort les valeurs de differential_evolution
delta, sigma, masse = rdm(L0, b, h, F, E)
print(f'kriging variance    b = {b:.1f} mm, h = {h:.1f} mm, masse = {masse:.2f} kg, delta reel = {delta:.3f} mm')

gp_delta, gp_sigma = entrainer(0, Xtr_n, Ytr), entrainer(1, Xtr_n, Ytr)

res_opt = differential_evolution(surrogate_kriging_smt, [borne_opti_b, borne_opti_h], args=(gp_delta, gp_sigma, sx))
b, h = res_opt.x # sort les valeurs de differential_evolution
delta, sigma, masse = rdm(L0, b, h, F, E)
print(f'smt sans marge      b = {b:.1f} mm, h = {h:.1f} mm, masse = {masse:.2f} kg, delta reel = {delta:.3f} mm')

#################
# ecriture des resultats en csv
#################
os.makedirs("resultats", exist_ok=True)
with open("resultats/florileche_resultat.csv", mode='w', encoding='utf8') as F:
    F.write('methode;sortie;R2;ERM\n')
    for methode, sortie, r_2, err in RESULTATS:
        F.write(f'{methode};{sortie};{r_2:.4f};{err:.2f}\n')