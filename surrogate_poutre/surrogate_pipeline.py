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
# paramétres données et probleme
SORTIES = ['delta', 'sigma']
L0 = 1500.0 
RESULTATS = []   # (methode, sortie, R2, ERM)
bornes_inf = np.array([1000.0, 20.0, 20.0])
bornes_supp = np.array([2000.0, 100.0, 100.0])
E = 210000 # MPA
fichier = 2000 #  N
delta_max = 5.0      # mm donné par la rdm
sigma_max= 200.0    # MPa donné par la rdm
l0 = 1500.0          # mm entre 1000 et 2000 ok
borne_opti_b , borne_opti_h = (bornes_inf[1],bornes_supp[1]) , (bornes_inf[2],bornes_supp[2])

# parametre polynomes
deg_poly = [ i for i in range(1,7)]

#tuning kriging
valeur_initiale_WK = 1e-4
bornes_wk = (1e-6, 1e-2)

theta0_smt= [1e-2]
theta_bounds_smt = [1e-4,1e0]

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
resu_poly_deg, Y_pred_log = resol_polynomes(Xtr, Ytr, Xtest, Xtr_n, Xtest_n,deg_poly)

for deg in deg_poly:
    res = evaluer(Ytest, resu_poly_deg[deg-1], f'poly{deg}')
    for s in SORTIES:
        r_2, err = res[s]               # on sort les deux métriques du tuple
        RESULTATS.append((f'poly{deg}', s, r_2, err))

res = evaluer(Ytest, Y_pred_log, 'poly_log')
for s in SORTIES:
    r_2, err = res[s]
    RESULTATS.append(('poly_log', s, r_2, err))

# par krigegage
Y_pred_gp, gp = resol_kriging(Xtr_n, Ytr, Xtest_n, valeur_initiale_WK, bornes_wk)   # gp resert pour l optimisation
res = evaluer(Ytest, Y_pred_gp, 'kriging')
for s in SORTIES:
    r_2, err = res[s]
    RESULTATS.append(('kriging', s, r_2, err))
 
# par krigegage smt
Yp = resol_kriging_smt(Xtr_n, Ytr, Ytest, Xtest_n, theta0_smt, theta_bounds_smt)
res = evaluer(Ytest, Yp, 'smt')
for s in SORTIES:
    r_2, err = res[s]
    RESULTATS.append(('smt', s, r_2, err))

#################
# utilisation : optimisation de la section
#################
# args= passe gp et sx a la fonction objectif, que differential_evolution
# n appelle qu avec le vecteur (b, h)

res_opt = differential_evolution(surrogate_kriging, [borne_opti_b, borne_opti_h], args=(gp, sx)) # on test les poutre
b, h = res_opt.x
delta, sigma, masse = rdm(L0, b, h, fichier, E)
print(f'kriging coef 0.95, b = {b:.1f} mm, h = {h:.1f} mm, masse = {masse:.2f} kg, delta reel = {delta:.3f} mm')

res_opt = differential_evolution(surrogate_kriging_variance, [borne_opti_b, borne_opti_h], args=(gp, sx))
b, h = res_opt.x # sort les valeurs de differential_evolution
delta, sigma, masse = rdm(L0, b, h, fichier, E)
print(f'kriging variance, b = {b:.1f} mm, h = {h:.1f} mm, masse = {masse:.2f} kg, delta reel = {delta:.3f} mm')

gp_delta, gp_sigma = entrainer(0, Xtr_n, Ytr), entrainer(1, Xtr_n, Ytr)

res_opt = differential_evolution(surrogate_kriging_smt, [borne_opti_b, borne_opti_h], args=(gp_delta, gp_sigma, sx))
b, h = res_opt.x # sort les valeurs de differential_evolution
delta, sigma, masse = rdm(L0, b, h, fichier, E)
print(f'smt sans marge, b = {b:.1f} mm, h = {h:.1f} mm, masse = {masse:.2f} kg, delta reel = {delta:.3f} mm')

#################
# ecriture des resultats en csv
#################

os.makedirs("resultats", exist_ok=True)
with open("resultats/florileche_resultat.csv", mode='w', encoding='utf8') as fichier:
    fichier.write('methode;sortie;R2;ERM\n')
    for methode, sortie, r_2, err in RESULTATS:
        fichier.write(f'{methode};{sortie};{r_2:.4f};{err:.2f}\n')