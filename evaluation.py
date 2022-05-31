from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import pandas as pd
import numpy as np
import os

def mdape(actuals, predicted):
    """
    Estimate MdAPE given actuals and predicted vectors
    """
    return np.median((np.abs(np.subtract(actuals, predicted)/ actuals)))

def evaluation(X_test, X_train, y_test, y_train, y_pred, regr, parameters, best_parameters, parameters_regr, model_label):
    """
    Evauates models based on MAE, MdAPE, MAPE and pricing deciles for MdAPE and MAPE
    """
    y_pred = np.exp(y_pred)
    y_test = np.exp(y_test)

    mape = mean_absolute_percentage_error(y_test, y_pred); print(f"MAPE: {round(mape, 3)}")
    mae = mean_absolute_error(y_test, y_pred); print(f"MAE: {round(mae, 1)}")
    mdape_ = mdape(y_test, y_pred); print(f"MdAPE: {round(mdape_, 3)} \n")

    X_output = X_test.copy()

    X_output['actuals'] = y_test
    X_output['predicted'] = y_pred

    X_output['decile'] = pd.qcut(X_output['actuals'], 10, labels = False)

    ### Evaluations at decile level
    mape_decile = []
    mdape_decile = []

    for i in range(10):
        temp = X_output[X_output["decile"] == i]
        mape_decile.append(mean_absolute_percentage_error(temp["actuals"], temp["predicted"]))
        mdape_decile.append(mdape(temp["actuals"], temp["predicted"]))

    label = parameters["label"]
    
    output_dir = f"output/{label}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_label in ["rf", "xgb"]: # Feature importance can only be estimated for rf and xgb
        importance = pd.Series(regr.feature_importances_ * 100, index = X_train.columns)
        importance = importance.sort_values(axis=0, ascending = False)
        importance.to_excel(f"{output_dir}/ft_importance_{model_label}.xlsx", index = True)
    elif model_label in ["glm"]:
        all_coefs = regr.summary()
    elif model_label in ["lasso", "ridge"]:
        all_coefs = pd.Series(regr.coef_.flatten(), index = X_train.columns)

    with open(f"{output_dir}/report_{model_label}.txt", "w", encoding="utf-8") as f:
        _ = f.write(str(parameters["label"]))
        _ = f.write('\n')
        _ = f.write(str(parameters["comment"]))
        _ = f.write('\n')
        _ = f.write('\n')

        _ = f.write(str(parameters["test_size"]))
        _ = f.write('\n')
        _ = f.write(str(parameters_regr))
        _ = f.write('\n')
        _ = f.write(f"Mape {mape}")
        _ = f.write('\n')
        _ = f.write(f"Mdape {mdape_}")
        _ = f.write('\n')
        _ = f.write(f"Mae {mae}")
        _ = f.write('\n')

        _ = f.write('Mape deciles')
        _ = f.write('\n')
        for k in mape_decile:
            _ = f.write(str(k))
            _ = f.write('\n')

        _ = f.write('Mdape deciles')
        _ = f.write('\n')

        for k in mdape_decile:
            _ = f.write(str(k))
            _ = f.write('\n')

        if model_label not in ["glm", "ridge", "lasso"]:
            _ = f.write(f"Best parms:")
            _ = f.write(str(best_parameters))

        if model_label == "glm":
            _ = f.write(f"Coefficients:")
            _ = f.write(str(all_coefs))       
        elif model_label in ["ridge", "lasso"]:
            all_coefs.to_excel(f"{output_dir}/report_{model_label}.xlsx", index = True)
