import pandas as pd
import numpy as np

class FileLoader:
    """
    Description : This class loads the excel file and prepares the data for further processing
    """
    def __init__(self, file_path, file_name):
        self.file_path = file_path
        self.file_name  = file_name
        if self.file_path == "" :
            self.file_address = self.file_name
        else :
            self.file_address = self.file_path + "\\" + self.file_name

    def load_file(self, classic_sheet_name, precarity_sheet_name, future = False):
        """
        Description : Loads the file input in file_path and file_name
        Arguments :
        - future (bool): Determines whether we load the futures data or not
        - classic_sheet_name (str): Name of the sheet containing classic data
        - precarity_sheet_name (str): Name of the sheet containing precarity data
        """
        # Loading the sheets
        classic_raw_df, precarity_raw_df = pd.read_excel(self.file_address, sheet_name = classic_sheet_name), pd.read_excel(self.file_address, sheet_name = precarity_sheet_name)

        # Formating the date column
        classic_raw_df["Date"] = pd.to_datetime(classic_raw_df["Date"], format="%Y-%m-%d")
        precarity_raw_df["Date"] = pd.to_datetime(precarity_raw_df["Date"], format="%Y-%m-%d")

        if not future :
            classic_raw_df, precarity_raw_df = classic_raw_df[["Date", "SPOT"]], precarity_raw_df[["Date", "SPOT"]]
            classic_raw_df.dropna(inplace=True)
            precarity_raw_df.dropna(inplace=True)

        
        classic_raw_df.fillna(method = "ffill", inplace=True)
        precarity_raw_df.fillna(method = "ffill", inplace=True)

        return classic_raw_df, precarity_raw_df
    
    def interpolate_dataframe(self, raw_data, columns):
        """
        Description : Interpolates the dataframe transforming the data from weekly to daily by doing linear interpolation on working days and adding some noise
        Arguments :
        - df (pd.DataFrame) : Dataframe of prices to interpolate
        - columns (list) :  the name of the columns to interpolate
        """
        df = raw_data.copy()
        # We go through the dataframe line by line ton interpolate on working days
        for i in range(1, len(df)):
            
            # Getting a list of business days between previous and next date
            start_date, end_date = df.loc[i-1, "Date"], df.loc[i, "Date"]
            business_days = pd.bdate_range(start=start_date, end=end_date).tolist()
            n_days = len(business_days)

            # Iterrating through columns to interpolate
            for column in columns :
                start_price, end_price = df.loc[i-1, column], df.loc[i, column]

                slope = (end_price - start_price) / (n_days - 1)
                intercept = start_price - slope * 1

                # Adding some incertitude
                mean, std = 0, abs(start_price - end_price)/2 # Standard deviation such that noise has 95% chance of being within start_price and end_price
                noise = np.random.normal(loc=mean, scale=std, size=n_days-2)
                
                for i, day in enumerate(business_days[1:-1]) :
                    interpolated_price = slope * (i + 1) + intercept + noise[i]
                    new_row = {"Date": day, column: interpolated_price}
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Clearing the dataframe
        df.sort_values(by="Date", inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        df.reset_index(drop=True, inplace=True)

        return df