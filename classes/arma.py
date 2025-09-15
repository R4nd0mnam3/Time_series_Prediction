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
