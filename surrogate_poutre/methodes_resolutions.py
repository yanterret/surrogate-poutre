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
def resol_polynomes(Xtr,Ytr,Xtest,Xtr_n,Xtest_n):
    '''
    entrée :
    Xtr,Ytr,Xtest,Xtr_n,Xtest_n : array
    Xtr et Xtest bruts : le log a besoin des vraies valeurs, les normalisées sont négatives
 
    sortie :
    resu_poly_deg : liste des 6 predictions, un array par degré
    Y_pred_log : array
    '''
    resu_poly_deg =[]
    for deg in [1, 2, 3, 4,5,6]:
        poly = make_pipeline(PolynomialFeatures(deg), LinearRegression())
        poly.fit(Xtr_n, Ytr)
        Ypred_poly = poly.predict(Xtest_n)
        resu_poly_deg.append(Ypred_poly,deg) # (2,)
    poly_log = make_pipeline(PolynomialFeatures(1), LinearRegression())
    poly_log.fit(np.log(Xtr), np.log(Ytr))
    Y_pred_log = np.exp(poly_log.predict(np.log(Xtest))) # ()
 
    return resu_poly_deg,Y_pred_log
 
#################
# méthodes de résolutions par kriging
#################
def resol_kriging(Xtr_n,Ytr,Xtest_n):
    '''
    entrée :
    Xtr_n,Ytr,Xtest_n : array
 
    sortie :
    Y_pred_gp : array
    gp : modèle entrainé
    '''
    kernel = C(1) * RBF(length_scale=np.ones(3)) + WhiteKernel(1e-4, (1e-6, 1e-2))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(Xtr_n, np.log(Ytr))
 
    Y_pred_gp = np.exp(gp.predict(Xtest_n)) # array (20, 2)
 
    return Y_pred_gp,gp
 
#################
# méthodes de résolutions par kriging smt
#################
def resol_kriging_smt(Xtr_n,Ytr,Ytest,Xtest_n):
    '''
    entrée :
    Xtr_n,Ytr,Ytest,Xtest_n : array
 
    sortie :
    Yp : array, Ytest sert uniquement à donner la forme
    '''
    Yp = np.zeros_like(Ytest)
    for j in SORTIES:
        sm = KRG(print_global=False,theta0=[1e-2],theta_bounds = [1e-4,1e0])
        sm.set_training_values(Xtr_n, np.log(Ytr[:, j]))
        sm.train()
        y =np.exp( sm.predict_values(Xtest_n)) # array (20, 1)
        Yp[:, j] = y.reshape((1,-1))   
    return Yp