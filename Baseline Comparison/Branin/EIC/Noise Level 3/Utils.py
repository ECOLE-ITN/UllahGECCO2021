import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.stats import norm
from smt.problems import Branin
from smt.surrogate_models import KRG
from smt.sampling_methods import LHS

def branin (X):
    """ 
    Branin Test Function. cf. Eq.(17)
    """
    x1, x2 = X.ravel()
    problem = Branin(ndim = 2)
    return problem(np.asarray([[x1],[x2]]).reshape(1,2))[0][0]                                                                         

def DOE(n_obs, xlimits, random_state = 0, criterion = 'm'):
    ''' 
    Latin HyperCube Sampling Design of Experiment for choosing initial sampling locations. cf. (Step 2 in Fig.2)
    '''
    sampling = LHS (xlimits = xlimits, random_state = random_state, criterion = criterion)
    return sampling(n_obs)


def noisy_prediction(Noise, Point, Model, Scaler):
    
    """
    For a Given point in the design space --- x --- and a Kriging model of the objective function,
    Returns Negative Kriging Prediction under additive Noise.
    """
    prediction = Model.predict(Scaler.transform(Point.reshape(1,-1) + Noise.reshape(1,-1))).reshape(1,-1)
    return - prediction.ravel()

def worst_case_prediction(Point , Model, Scaler):
    
    """ 
    For a Given point in the design space --- x --- and a Kriging model of the objective function, 
    find the worst Kriging prediction for x in the face of uncertainty. cf. Eq.(9)
    ----------
        Point: array-like, shape = [1, n_dimensions]
            The point in the design domain for which the worst Kriging needs to be computed.
        Model: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated samples.
    """
    seed = 0
    worst_noise = None
    worst_function_value = np.inf
    # Bounds of Noise
    
    bounds = ((-3,3),(-3,3))
    n_restarts = 2
    n_params = 2
    
    np.random.seed(seed)
    for starting_point in np.random.uniform(np.array(bounds)[:,0], np.array(bounds)[:,1], size = (n_restarts, n_params)):
        res = minimize(fun = noisy_prediction,
                       x0 = starting_point.reshape(n_params,-1),
                       bounds = bounds,
                       method = 'L-BFGS-B',
                       args = (Point.reshape(1,-1) , Model, Scaler))
        
        if res.fun < worst_function_value:
            worst_function_value = res.fun
            worst_noise = res.x
           
    return Model.predict(Scaler.transform(Point.reshape(1,-1) + worst_noise.reshape(1,-1))).reshape(1,-1).ravel()

def noise(Point , Model, Scaler):
    
    """ 
    For a Given point in the design space --- x --- and a Kriging model of the objective function, 
    find worst noise value, i.e., the one which realizes the worst Kriging prediction. cf. Eq.(10)
    ----------
        Point: array-like, shape = [1, n_dimensions]
            The point in the design domain for which the noise needs to be computed.
        Model: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated samples.
    """
    seed = 0
    worst_noise = None
    worst_function_value = np.inf
    # Bounds of Noise
    bounds = ((-3,3),(-3,3))
    #bounds = np.array([[-0.125,-0.125,-0.125],[0.125,0.125,0.125]])
    n_restarts = 2
    n_params = 2
    
    np.random.seed(seed)
    for starting_point in np.random.uniform(np.array(bounds)[:,0], np.array(bounds)[:,1], size=(n_restarts, n_params)):
        res = minimize(fun = noisy_prediction,
                       x0 = starting_point.reshape(n_params,-1),
                       bounds = bounds,
                       method = 'L-BFGS-B',
                       args = (Point.reshape(1,-1), Model, Scaler))
        
        if res.fun < worst_function_value:
            worst_function_value = res.fun
            worst_noise = res.x
        
    
    return worst_noise

def expected_improvement(candidate, robust_optimum, Model, Scaler, n_params = 2):
    
    """ expected_improvement
    Expected improvement acquisition function. cf. Eq.(15)
    Arguments:
    ----------
        candidate: array-like, shape = [1, n_dimensions]
            The point for which the expected improvement needs to be computed.
        Model: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated samples.
        robust_optimum: array-like, shape = [1, n_dimensions]
            Numpy array that contains the value of the robust optimum found so far 
               n_params: int.
            Dimension of the design space.
    """
    
    
    worst_noise = noise (candidate, Model, Scaler)
    #candidate = Scaler.transform (candidate)
    
    mu, sigma = Model.predict(Scaler.transform(candidate.reshape(1,-1) + worst_noise.reshape(1,-1)), return_std = True)
    #sigma = np.sqrt(Model.predict_variances(candidate.reshape(1,-1) + worst_noise.reshape(1,-1)))
    
    
    # In case sigma equals zero
    with np.errstate(divide = 'ignore'):
        Z = ( robust_optimum - mu) / sigma
        expected_improvement = (robust_optimum - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement [sigma == 0.0] == 0.0
        
    return - expected_improvement.ravel()


def sample_next_point(acquisition_func, Model, robust_optimum, Scaler, bounds = ((-2,7),(3,12)) , n_restarts = 25):
    
    """ sample_next_location
    Proposes the next location to sample for the objective function. cf. (Step 6 in Fig.2)
    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        Model: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated function responses.
        robust_optimum: array-like, shape = [1, n_dimensions]
            Numpy array that contains the values off the robust optimum found so far
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.
    """
    seed = 0
    best_x = None
    best_acquisition_value = np.inf
    n_params = 2
    
    np.random.seed(seed)
    for starting_point in np.random.uniform(np.array(bounds)[:,0], np.array(bounds)[:,1], size = (n_restarts, n_params)):
        res = minimize(fun = acquisition_func,
                       x0 = starting_point.ravel(),
                       bounds = bounds,
                       method = 'L-BFGS-B',
                       args = (robust_optimum, Model, Scaler, n_params))
        
        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x
        
    return best_x.ravel()
    
    
def find_robust_optimum (Model, Scaler):
    
    """ 
    Find the reference robust optimum, i.e., By minimizing the Worst Case
    Kriging Prediction w.r.t. the Uncertainty Set. cf. Eq.(8)
    """
    seed = 0
    best_x = None
    best_function_value = np.inf
    bounds = ((-2,7),(3,12))
    n_params = 2
    n_restarts = 2
    
    np.random.seed(seed)
    for starting_point in np.random.uniform(np.array(bounds)[:,0], np.array(bounds)[:,1], size = (n_restarts, n_params)):
        res = minimize(fun = worst_case_prediction,
                       x0 = starting_point.reshape(n_params,-1),
                       bounds = bounds,
                       method = 'L-BFGS-B',
                       args = (Model, Scaler))
        
        if res.fun < best_function_value:
            best_function_value = res.fun
            best_x = res.x
    
    
    return best_function_value