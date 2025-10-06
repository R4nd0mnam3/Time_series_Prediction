import numpy as np
from sklearn.metrics import mean_squared_error

class train_test_split:
    def __init__(self, dependent_time_series, train_test_ratio=None, split_index=None):
        self.dependent_time_series = dependent_time_series
        self.train_test_ratio = train_test_ratio
        self.split_index = split_index

    def train_test_split(self):
        """
        Description : Split the data between train and test depedning on either train_test_ratio or split_index
        """
        if self.split_index is None and self.train_test_ratio is None :
            raise ValueError("Either train_test_ratio or split_index must be provided.")    
        
        elif self.train_test_ratio is not None:
            split_index = int(len(self.dependent_time_series) * self.train_test_ratio)
            self.train_dependent = self.dependent_time_series[:split_index]
            self.test_dependent = self.dependent_time_series[split_index:].reset_index(drop=True)

        elif self.split_index is not None:
            self.train_dependent = self.dependent_time_series[:self.split_index]
            self.test_dependent = self.dependent_time_series[self.split_index:].reset_index(drop=True)

class model_metrics:
    def __init__(self, train_real, train_pred, test_real, test_pred):
        self.train_real = train_real
        self.train_pred = train_pred
        self.test_real = test_real
        self.test_pred = test_pred

    def mae(self):
        """
        Description : Computes the MAE for both train and test sets
        """
        train_mae = np.mean(np.abs(self.train_real - self.train_pred))
        test_mae = np.mean(np.abs(self.test_real - self.test_pred))

        return float(train_mae), float(test_mae)

    def rmse(self):
        """
        Description : Computes the RMSE for both train and test sets
        """
        train_rmse = np.sqrt(mean_squared_error(self.train_real, self.train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.test_real, self.test_pred))

        return float(train_rmse), float(test_rmse)
    
    def mape(self):
        """
        Description : Computes the MAPE for both train and test sets
        """
        train_mape = np.mean(np.abs((self.train_real - self.train_pred) / self.train_real)) * 100
        test_mape = np.mean(np.abs((self.test_real - self.test_pred) / self.test_real)) * 100

        return float(train_mape), float(test_mape)
    
    def r2(self):
        """
        Description : Computes the R2 for both train and test sets
        """
        train_ss_res = np.sum((self.train_real - self.train_pred) ** 2)
        train_ss_tot = np.sum((self.train_real - np.mean(self.train_real)) ** 2)
        train_r2 = 1 - (train_ss_res / train_ss_tot)

        test_ss_res = np.sum((self.test_real - self.test_pred) ** 2)
        test_ss_tot = np.sum((self.test_real - np.mean(self.test_real)) ** 2)
        test_r2 = 1 - (test_ss_res / test_ss_tot)

        return float(train_r2), float(test_r2)
    
    def get_all_metrics(self):
        """
        Description : Computes all metrics for both train and test sets
        """
        train_mae, test_mae = self.mae()
        train_rmse, test_rmse = self.rmse()
        train_mape, test_mape = self.mape()
        train_r2, test_r2 = self.r2()

        return {
            "train": {
                "mae": train_mae,
                "rmse": train_rmse,
                "mape": train_mape,
                "r2": train_r2
            },
            "test": {
                "mae": test_mae,
                "rmse": test_rmse,
                "mape": test_mape,
                "r2": test_r2
            }
        }