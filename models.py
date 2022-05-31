from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import statsmodels.api as sm
import sklearn.linear_model as skl_lm
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, GridSearchCV
import multiprocessing
import pandas as pd
from evaluation import *


def run_xgboost(parameters, X_train, y_train, X_test, y_test):
    """
    Run the xgboost pipeline and its hypetparameter tuning
    """
    parameters_regr = parameters["parameters_xgb"]

    model_xgb = GradientBoostingRegressor(random_state = parameters["random_state"])

    k_fold = KFold(n_splits = parameters["kf"], shuffle=True, random_state = parameters["random_state"])

    n_jobs = multiprocessing.cpu_count()-1

    grid = GridSearchCV(model_xgb,
                            parameters_regr, 
                            refit = False,
                            cv = k_fold,
                            n_jobs = n_jobs,
                            scoring = "neg_mean_absolute_percentage_error")

    grid.fit(X_train, y_train)

    best_parameters = grid.best_params_

    regr = GradientBoostingRegressor(**best_parameters, random_state = parameters["random_state"])
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    # Evaluation
    evaluation(X_test, X_train, y_test, y_train, y_pred, regr, parameters, best_parameters, parameters_regr, "xgb")

def run_rf(parameters, X_train, y_train, X_test, y_test):
    """
    Run the random forest pipeline and its hypetparameter tuning
    """
    parameters_regr = parameters["parameters_rf"]

    model = RandomForestRegressor(random_state = parameters["random_state"])

    k_fold = KFold(n_splits = parameters["kf"], shuffle=True, random_state = parameters["random_state"])

    n_jobs = multiprocessing.cpu_count()-1

    grid = GridSearchCV(model,
                            parameters_regr, 
                            refit = False,
                            cv = k_fold,
                            n_jobs = n_jobs,
                            scoring = "neg_mean_absolute_percentage_error")

    grid.fit(X_train, y_train)

    best_parameters = grid.best_params_

    regr = RandomForestRegressor(**best_parameters, random_state = parameters["random_state"])
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    # Evaluation
    evaluation(X_test, X_train, y_test, y_train, y_pred, regr, parameters, best_parameters, parameters_regr, "rf")

def run_svm(parameters, X_train, y_train, X_test, y_test):
    """
    Run the svm pipeline and its hypetparameter tuning
    """
    parameters_regr = parameters["parameters_svm"]

    model = SVR()

    k_fold = KFold(n_splits = parameters["kf"], shuffle=True, random_state = parameters["random_state"])

    n_jobs = multiprocessing.cpu_count()-1

    grid = GridSearchCV(model,
                            parameters_regr, 
                            refit = False,
                            cv = k_fold,
                            n_jobs = n_jobs,
                            scoring = "neg_mean_absolute_percentage_error")

    scaler = StandardScaler().fit(X_train)

    grid.fit(scaler.fit_transform(X_train), y_train)

    best_parameters = grid.best_params_

    regr = SVR(**best_parameters)
    regr.fit(scaler.fit_transform(X_train), y_train)

    y_pred = regr.predict(scaler.fit_transform(X_test))

    # Evaluation
    evaluation(X_test, X_train, y_test, y_train, y_pred, regr, parameters, best_parameters, parameters_regr, "svm")

def run_nnet(parameters, X_train, y_train, X_test, y_test):
    """
    Run the mlp regressor pipeline and its hypetparameter tuning
    """
    parameters_regr = parameters["parameters_nnet"]

    model = MLPRegressor(random_state = parameters["random_state"])

    k_fold = KFold(n_splits = parameters["kf"], shuffle=True, random_state = parameters["random_state"])

    n_jobs = multiprocessing.cpu_count()-1

    grid = GridSearchCV(model,
                            parameters_regr, 
                            refit = False,
                            cv = k_fold,
                            n_jobs = n_jobs,
                            scoring = "neg_mean_absolute_percentage_error")

    grid.fit(X_train, y_train)

    best_parameters = grid.best_params_

    regr = MLPRegressor(**best_parameters, random_state = parameters["random_state"])
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    # Evaluation
    evaluation(X_test, X_train, y_test, y_train, y_pred, regr, parameters, best_parameters, parameters_regr, "nnet")

def run_glm(parameters, X_train, y_train, X_test, y_test):
    """
    Run the glm pipeline
    """
    model = sm.OLS(y_train, X_train)
    regr = model.fit()

    y_pred = regr.predict(X_test)

    evaluation(X_test, X_train, y_test, y_train, y_pred, regr, parameters, None, None, "glm")

def run_lasso(parameters, X_train, y_train, X_test, y_test):
    """
    Run the lasso pipeline
    """
    alphas = 10**np.linspace(parameters["lasso_alpha_initial"], parameters["lasso_alpha_final"], parameters["lasso_linspace"])
    
    scaler = StandardScaler().fit(X_train)

    lassocv = skl_lm.LassoCV(alphas=alphas, max_iter = parameters["lasso_max_iter"], random_state = parameters["random_state"])

    lassocv.fit(scaler.fit_transform(X_train), y_train.values.ravel())

    optimal_lassocv = skl_lm.Lasso(random_state = parameters["random_state"])
    optimal_lassocv.set_params(alpha= lassocv.alpha_)

    optimal_lassocv.fit(scaler.fit_transform(X_train), y_train.values.ravel())

    y_pred = optimal_lassocv.predict(scaler.fit_transform(X_test))

    evaluation(X_test, X_train, y_test, y_train, y_pred, optimal_lassocv, parameters, None, None, "lasso")

def run_ridge(parameters, X_train, y_train, X_test, y_test):
    """
    Run the ridge pipeline
    """
    alphas = 10**np.linspace(parameters["ridge_alpha_initial"], parameters["ridge_alpha_final"], parameters["ridge_linspace"])
    
    scaler = StandardScaler().fit(X_train)

    ridgecv = skl_lm.RidgeCV(alphas=alphas)

    ridgecv.fit(scaler.fit_transform(X_train), y_train.values.ravel())

    optimal_ridgecv = skl_lm.Ridge(random_state = parameters["random_state"])
    optimal_ridgecv.set_params(alpha= ridgecv.alpha_)

    optimal_ridgecv.fit(scaler.fit_transform(X_train), y_train.values.ravel())

    y_pred = optimal_ridgecv.predict(scaler.fit_transform(X_test))

    evaluation(X_test, X_train, y_test, y_train, y_pred, optimal_ridgecv, parameters, None, None, "ridge")