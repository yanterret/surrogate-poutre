
import numpy as np 
from scipy.optimize import differential_evolution

DELTA_MAX = 5.0      # mm on le pose pour satisfaire la rdm
SIGMA_MAX = 200.0    # MPa donné par la rdm
L0 = 1500.0          # mm entre 1000 et 2000 ok
RHO = 7850e-9        # kg/mm3 acier

def surrogate_kriging(y):
    b, h = y
    delta, sigma = np.exp(gp.predict(sx.transform([[L0, b, h]]))[0])   # fleche, contrainte
    if delta <= DELTA_MAX and sigma <= SIGMA_MAX: # savoir si la poutre respecte les conditions physiques  
        return RHO * b * h * L0      # score normal 
    return 1e10 # conditions physiques élevée on fait augmenter le score de maniere absurde ( on en veut pas )

res = differential_evolution(surrogate_kriging, [(20, 100), (20, 100)]) # on test les poutre 
b, h = res.x
b,h
