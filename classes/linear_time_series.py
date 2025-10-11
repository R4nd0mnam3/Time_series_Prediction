import numpy as np
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

import classes.tools as tools

class  LinearTimeSeriesModel(tools.train_test_split):
    def __init__(self, dependent_time_series, train_test_ratio = 0.8):
        super().__init__(dependent_time_series, train_test_ratio)
        self.train_test_split()

    def get_ar_max_order(self, max_lag=10):
        """
        Description : Selects the maximum order of the AR model using PACF cutoff method
        Argumments:
        - max_lag (int) : Maximum number of lags to consider
        """
        pacf_vals, confint = pacf(self.train_dependent, nlags=max_lag, alpha=0.05, method="yw")
        order = 0

        for lo, hi in confint[1:]:
            if hi<0 or lo>0:
                order += 1
            else:
                break

        self.ar_max_order = order

    def get_ma_max_order(self, max_lag=10):
        """
        Description : Selects the maximum order of the MA model using ACF cutoff method
        Argumments:
        - max_lag (int) : Maximum number of lags to consider
        """
        acf_vals, confint = acf(self.train_dependent, nlags=max_lag, alpha=0.05, fft=True)
        order = 0
        
        for lo, hi in confint[1:]:
            if hi<0 or lo>0:
                order += 1
            else:
                break
        
        self.ma_max_order = order
    
    def get_model(self, series, ma_order, ar_order, integ=0):
        """
        Description : Fits an ARIMA model to the time series given the MA and AR orders
        Arguments:
        - series (pd.series(float)) : Time series to fit
        - ma_order (int) : Order of the MA model
        - ar_order (int) : Order of the AR model
        """
        model = ARIMA(series, order=(ar_order, integ, ma_order))
        model_fit = model.fit()

        return model_fit

    def aicc(self, aic, k, n):
        """
        Description : Computes the AICc value given the AIC value, number of parameters and number of observations
        Arguments:
        - aic (float) : AIC value
        - k (int) : Number of parameters
        - n (int) : Number of observations
        """
        return float(aic + (2*k*(k+1))/(n-k-1))
    
    def select_model(self, ljung_lags=[15,15], alpha=0.05):
        """
        Description : Selects the best ARIMA model using the AICC criteria
        Arguments:
        - ljung_lags (int) : Number of lags to consider for the residuals analysis
        - alpha (float) : Significance level for the Ljung-Box test
        """
        best_aicc = np.inf
        best_aicc_order = None
        best_aicc_model = None
        
        for ma_order in range(self.ma_max_order + 1):
            for ar_order in range(self.ar_max_order + 1):
                model_fit = self.get_model(self.train_dependent, ma_order, ar_order)
                aic = model_fit.aic
                aicc = self.aicc(aic, model_fit.k_params, len(self.train_dependent))

                # We conduct Ljung_Box test to see if the residuals are white noise
                residuals = model_fit.resid
                lb = acorr_ljungbox(residuals, lags=ljung_lags, return_df=True)
                lb_pvalue_min = float(lb["lb_pvalue"].min())
                print(f"The p-value of the L-JungBox test is {lb_pvalue_min} for the model ARIMA({ar_order},0,{ma_order}) with AIC = {aic}")

                # Null hypothesis : the residuals are white noise
                if lb_pvalue_min > alpha: # If pvalue > alpha we accept the null hypothesis and the model is valid
                    if aicc < best_aicc:
                        best_aicc = aicc
                        best_aicc_order = (ar_order, 0, ma_order)
                        best_aicc_model = model_fit

        self.model = best_aicc_model
        return {
            "aicc" : {
                "order": best_aicc_order,
                "aicc": best_aicc
            }}
    
    def model_prediction(self, start_index=None, end_index=None):
        """
        Descritption : Returns the prediction for both the training and testing sets
        Arguments :
        - start_index (int) : Start index for the prediction
        - end_index (int) : End index for the prediction
        """
        # In case no indices are provided we predict on the whole dataset
        if start_index is None and end_index is None :
            start_index = 0
            end_index = len(self.dependent_time_series)
            prediction = self.model.predict(start=start_index, end=end_index-1)
            train_pred = prediction[:len(self.train_dependent)]
            validation_pred = prediction[len(self.train_dependent):]

            return train_pred, validation_pred
        
        # In case both indices are provided we predict on the given range
        else :
            prediction = self.model.predict(start=start_index, end=end_index)

            return prediction