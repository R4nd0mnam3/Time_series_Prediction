import numpy as np
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

class  LineaTimeSeriesModel:
    def __init__(self, dependent_time_series, train_test_ratio = 0.8):
        self.dependent_time_series = dependent_time_series
        self.train_test_ratio = train_test_ratio
    
    def train_test_split(self):
        """
        Description : Split the data between train and test
        """
        split_index = int(len(self.dependent_time_series) * self.train_test_ratio)
        self.train_dependent = self.dependent_time_series[:split_index]
        self.test_dependent = self.dependent_time_series[split_index:]
        
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
        
        return int(order)


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
        model_fit = model.fit()

        return model_fit
    
    def select_model(self, max_ma_order, max_ar_order, ljung_lags=(15,15), alpha=0.05):
        """
        Description : Selects the best ARIMA model using the AIC criteria
        Arguments:
        - max_ma_order (int) : Maximum order of the MA model to consider
        - max_ar_order (int) : Maximum order of the AR model to consider
        - ljung_lags (int) : Number of lags to consider for the residuals analysis
        - alpha (float) : Significance level for the Ljung-Box test
        """
        best_aic = np.inf
        best_aic_order = None
        best_aic_model = None
        
        for ma_order in range(max_ma_order + 1):
            for ar_order in range(max_ar_order + 1):
                model_fit = self.get_model(self.train_dependent, ma_order, ar_order)
                aic = model_fit.aic

                # We conduct Ljung_Box test to see if the residuals are white noise
                residuals = model_fit.resid
                lb = acorr_ljungbox(residuals, lags=ljung_lags, return_df=True)
                lb_pvalue_min = float(lb["lb_pvalue"].min())

                # Null hypothesis : the residuals are white noise
                if lb_pvalue_min > alpha: # If pvalue > alpha we accept the null hypothesis and the model is valid
                    if aic < best_aic:
                        best_aic = aic
                        best_aic_order = (ar_order, 0, ma_order)
                        best_aic_model = model_fit

        return {
            "aic": {
                "order": best_aic_order,
                "model": best_aic_model,
                "aic": best_aic
            }}
    
    def model_prediction(self, model):
        """
        Descritption : Returns the prediction for both the training and testing sets
        - model : Fitted ARIMA model
        Arguments: 
        - model (ARIMA) : Fitted ARIMA model
        """
        start_index = 0
        end_index = len(self.dependent_time_series)
        prediction = model.predict(start=start_index, end=end_index-1)
        train_pred = prediction[:len(self.train_dependent)]
        test_pred = prediction[len(self.train_dependent):]

        return train_pred, test_pred