#################
# Libraries
#################
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from smt.surrogate_models import KRG
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import numpy as np

SORTIES = ['delta', 'sigma']

#################
# méthodes de résolutions par polynomes et log polynomes
#################
def resol_polynomes(Xtr,Ytr,Xtest,Xtr_n,Xtest_n,deg_poly):
    '''
    entrée :
    Xtr,Ytr,Xtest,Xtr_n,Xtest_n : array
 
    sortie :
    resu_poly_deg : liste : un array par degré
    Y_pred_log : array
    '''
    resu_poly_deg =[]
    for deg in deg_poly:
        poly = make_pipeline(PolynomialFeatures(deg), LinearRegression())
        poly.fit(Xtr_n, Ytr)
        Ypred_poly = poly.predict(Xtest_n)
        resu_poly_deg.append(Ypred_poly) # un array (20,2) par degré
    poly_log = make_pipeline(PolynomialFeatures(1), LinearRegression())
    poly_log.fit(np.log(Xtr), np.log(Ytr))
    Y_pred_log = np.exp(poly_log.predict(np.log(Xtest))) # array (20, 2)
 
    return resu_poly_deg,Y_pred_log
 
#################
# méthodes de résolutions par kriging
#################
def resol_kriging(Xtr_n,Ytr,Xtest_n,valeur_initiale_WK,bornes_wk):
    '''
    entrée :
    Xtr_n,Ytr,Xtest_n : array
 
    sortie :
    Y_pred_gp : array
    gp : modèle entrainé
    '''
    kernel = C(1) * RBF(length_scale=np.ones(3)) + WhiteKernel(valeur_initiale_WK, bornes_wk)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(Xtr_n, np.log(Ytr))
 
    Y_pred_gp = np.exp(gp.predict(Xtest_n)) # array (20, 2)
 
    return Y_pred_gp,gp
 
#################
# méthodes de résolutions par kriging smt
#################
def resol_kriging_smt(Xtr_n,Ytr,Ytest,Xtest_n,theta0_smt,theta_bounds_smt):
    '''
    entrée :
    Xtr_n,Ytr,Ytest,Xtest_n : array
 
    sortie :
    Yp : array, Ytest sert uniquement à donner la forme
    '''
    Yp = np.zeros_like(Ytest)
    for j in [0, 1]:   # 0 = delta, 1 = sigma 
        sm = KRG(print_global=False,theta0=theta0_smt,theta_bounds = theta_bounds_smt)
        sm.set_training_values(Xtr_n, np.log(Ytr[:, j]))
        sm.train()
        y =np.exp( sm.predict_values(Xtest_n)) # array (20, 1)
        Yp[:, j] = y.reshape((1,-1))   
    return Yp