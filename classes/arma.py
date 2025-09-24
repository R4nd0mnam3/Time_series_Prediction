import numpy as np
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

class ARMA :
    def __init__(self, dependent_time_series, train_test_ratio = 0.8, arimax = False, explanatory_time_series = None):
        self.arimax = arimax
        self.dependent_time_series = dependent_time_series
        self.explanatory_time_series = explanatory_time_series
        self.train_test_ratio = train_test_ratio
    
    def train_test_split(self):
        """
        Description : Split the data between train and test
        """
        split_index = int(len(self.dependent_time_series) * self.train_test_ratio)
        self.train_dependent = self.dependent_time_series[:split_index]
        self.test_dependent = self.dependent_time_series[split_index:]
        
        if self.arimax and self.explanatory_time_series is not None:
            self.train_explanatory = self.explanatory_time_series[:split_index]
            self.test_explanatory = self.explanatory_time_series[split_index:]

    def get_ar_max_order(self, max_lag=10):
        """
        Description : Selects the maximum order of the AR model using PACF cutoff method
        Argumments:
        - series (pd.series(float)) : Time series to analyze
        - max_lag (int) : Maximum number of lags to consider
        """
        pacf_vals, confint = pacf(self.train_dependent, nlags=max_lag, alpha=0.05, method="yw")
        print(pacf_vals)
        print(confint)
        order = 0
        for v, (lo, hi) in zip(pacf_vals[1:], confint[1:]):
            if v < lo or v > hi:
                order += 1
            else:
                break
        
        return int(order)


    def get_ma_max_order(self, max_lag=10):
        """
        Description : Selects the maximum order of the MA model using ACF cutoff method
        Argumments:
        - series (pd.series(float)) : Time series to analyze
        - max_lag (int) : Maximum number of lags to consider
        """
        acf_vals, confint = acf(self.train_dependent, nlags=max_lag, alpha=0.05, fft=True)
        print(acf_vals)
        print(confint)
        order = 0
        for v, (lo, hi) in zip(acf_vals[1:], confint[1:]):
            if v < lo or v > hi:
                order += 1
            else:
                break

        return int(order)
    
    def get_model(self, series, ma_order, ar_order, integ=0):
        """
        Description : Fits an ARIMA model to the time series given the MA and AR orders
        Arguments:
        - series (pd.series(float)) : Time series to fit
        - ma_order (int) : Order of the MA model
        - ar_order (int) : Order of the AR model
        """
        model = ARIMA(series, order=(ar_order, integ, ma_order))
        self.model_fit = model.fit()
    
    def select_model(self, max_ma_order, max_ar_order, ljung_lags=(15,15), alpha=0.05):
        """
        Description : Selects the best ARIMA model using the AIC and BIC criterions
        Arguments:
        - series (pd.series(float)) : Time series to fit
        - max_ma_order (int) : Maximum order of the MA model to consider
        - max_ar_order (int) : Maximum order of the AR model to consider
        - ljung_lags (int) : Number of lags to consider for the residuals analysis
        - alpha (float) : Significance level for the Ljung-Box test
        """
        best_aic = np.inf
        best_bic = np.inf
        best_aic_order = None
        best_bic_order = None
        best_aic_model = None
        best_bic_model = None
        
        for ma_order in range(max_ma_order + 1):
            for ar_order in range(max_ar_order + 1):
                model_fit = ARMA.get_model(self.train_dependent, ma_order, ar_order)
                aic = model_fit.aic
                bic = model_fit.bic
                residuals = model_fit.resid

                # We conduct Ljung_Box test to see if the residuals are white noise
                lb = acorr_ljungbox(residuals, lags=ljung_lags, return_df=True)
                lb_pvalue_min = float(lb["lb_pvalue"].min())
                # Null hypothesis : the residuals are white noise
                if lb_pvalue_min > alpha: # If pvalue > alpha we accept the null hypothesis and the model is valid
                    if aic < best_aic:
                        best_aic = aic
                        best_aic_order = (ar_order, 0, ma_order)
                        best_aic_model = model_fit
                    
                    if bic < best_bic:
                        best_bic = bic
                        best_bic_order = (ar_order, 0, ma_order)
                        best_bic_model = model_fit
                else :
                    return None

        return {
            "aic": {
                "order": best_aic_order,
                "model": best_aic_model,
                "aic": best_aic
            },
            "bic": {
                "order": best_bic_order,
                "model": best_bic_model,
                "bic": best_bic
            }
        }