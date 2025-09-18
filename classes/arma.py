import numpy as np
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.tsa.arima.model import ARIMA

class ARMA :
    def __init__(self, dependent_time_series, train_test_ratio = 0.8, arimax = False, explanatory_time_series = None):
        self.arimax = arimax
        self.dependent_time_series = dependent_time_series
        self.explanatory_time_series = explanatory_time_series
        self.train_test_ratio = train_test_ratio
    
    def train_test_split(self):
        split_index = int(len(self.dependent_time_series) * self.train_test_ratio)
        self.train_dependent = self.dependent_time_series[:split_index]
        self.test_dependent = self.dependent_time_series[split_index:]
        
        if self.arimax and self.explanatory_time_series is not None:
            self.train_explanatory = self.explanatory_time_series[:split_index]
            self.test_explanatory = self.explanatory_time_series[split_index:]

    def get_ar_order(series, max_lag=10):
        """
        Description : Selects the maximum order of the AR model using PACF cutoff method
        Argumments:
        - series (pd.series(float)) : Time series to analyze
        - max_lag (int) : Maximum number of lags to consider
        """
        pacf_vals, confint = pacf(y, nlags=max_lag, alpha=0.05, method="yw")
        
        order = 0
        for v, (lo, hi) in zip(pacf_vals[1:], confint[1:]):
            if v < lo or v > hi:
                order += 1
            else:
                break
        return int(order)


    def get_ma_order(series, max_lag=10):
        """
        Description : Selects the maximum order of the AR model using PACF cutoff method
        Argumments:
        - series (pd.series(float)) : Time series to analyze
        - max_lag (int) : Maximum number of lags to consider
        """
        acf_vals, confint = acf(y, nlags=max_lag, alpha=0.05, fft=True)
        
        order = 0
        for v, (lo, hi) in zip(acf_vals[1:], confint[1:]):
            if v < lo or v > hi:
                order += 1
            else:
                break
        return int(order)
    

    def select_arma_orders(series, max_p=5, max_q=5, exog=None, trend="c"):

    ic_rows = []

    best_aic = np.inf
    best_aic_order = (0, 0)
    best_bic = np.inf
    best_bic_order = (0, 0)

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            # Skip the trivial (0,0) if you don't want white noise; keep it here for completeness
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(endog=y, exog=exog, order=(p, 0, q), trend=trend, enforce_stationarity=False, enforce_invertibility=False)
                    fit = model.fit()
                aic = fit.aic
                bic = fit.bic
                converged = True
            except Exception:
                aic = np.inf
                bic = np.inf
                converged = False

            ic_rows.append({"p": p, "q": q, "aic": aic, "bic": bic, "converged": converged})

            if aic < best_aic:
                best_aic = aic
                best_aic_order = (p, q)
            if bic < best_bic:
                best_bic = bic
                best_bic_order = (p, q)

    return {
        "best_aic_order": best_aic_order,
        "best_aic": best_aic,
        "best_bic_order": best_bic_order,
        "best_bic": best_bic,
        "ic_table": ic_rows,
    }