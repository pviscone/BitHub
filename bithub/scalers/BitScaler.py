import numpy as np
import pandas as pd

class BitScaler:
    """
    A class for scaling numerical data using bit scaling. Like min-max but dividing with the smallest power of 2 that covers the range.
    Methods:
    - __init__(): Initializes the BitScaler object.
    - auto_range(df, columns=None): Calculates the range of values for the specified columns in the given DataFrame.
    - fit(range_dict=None, target=(-1, 1)): Fits the scaler to the given range dictionary and target values.
    - apply(df): Applies the scaling functions to the specified columns of the given DataFrame.
    - save(filename): Save the scaler to a file.
    - load(filename): Load the scaler parameters from a file and fit the scaler.
    """

    def __init__(self) -> None:
        """
        Initializes the BitScaler object.
        """
        self.fitted = False
        self.range_dict = None
        self.scale_funcs = {}
        self.bit_shifts = {}
        self.df = None

    def fit(self, df, columns=None, range_dict=None, target=(-1, 1)):
        """
        Calculates the range of values for the specified columns in the given DataFrame and fits the scaler to the given range dictionary and target values.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
            columns (list, optional): The list of column names to calculate the range for. If not provided, the range will be calculated for all columns in the DataFrame.
            range_dict (dict|None, optional): A dictionary containing the range values for each key. Can be None if the range is to be automatically calculated with auto_range. Defaults to None.
            target (tuple, optional): A tuple containing the new low and high values for scaling. Defaults to (-1, 1).
        Raises:
            ValueError: If the scaler is already fitted.
        Returns:
            None
        """
        if self.fitted:
            raise ValueError("Scaler already fitted")

        columns = df.columns if columns is None else columns

        if range_dict is not None:
            self.range_dict = range_dict
        else:
            self.range_dict = {key: (df[key].min(), df[key].max()) for key in columns}

        inf, sup = target

        for key in self.range_dict:
            min_x, max_x = self.range_dict[key]

            self.bit_shifts[key] = np.ceil(np.log2((max_x - min_x)/(sup - inf)))

            #! God damn, python, you suck!!!
            #! Dirty workaround to avoid the lambda function to capture the last value of the loop
            self.scale_funcs[key] = eval(
                f"lambda x: {inf}+((x-{min_x}))/(2**{self.bit_shifts[key]})"
            )

        self.target = target
        self.fitted = True
        self.df = self.get_df()

    def apply(self, df, copy = True):
        """
        Applies the scaling functions to the specified columns of the given DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to apply the scaling functions to.

        Returns:
            pandas.DataFrame: The DataFrame with the scaled columns.

        Raises:
            ValueError: If the scaler has not been fitted.
        """
        if copy:
            df = df.copy()
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        for key in self.scale_funcs:
            df[key] = self.scale_funcs[key](df[key])
        return df

    def get_df(self):
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        if self.df is None:
            df_dict = {"feature_name": [], "inf": [], "sup": [], "min": [], "max": [], "bit_shift": []}
            for key in self.range_dict:
                df_dict["feature_name"].append(key)
                df_dict["inf"].append(self.target[0])
                df_dict["sup"].append(self.target[1])
                df_dict["min"].append(self.range_dict[key][0])
                df_dict["max"].append(self.range_dict[key][1])
                df_dict["bit_shift"].append(int(self.bit_shifts[key]))
            self.df = pd.DataFrame(df_dict)
            return self.df
        else:
            return self.df


    def save(self, filename):
        """
        Save the scaler to a file.

        Args:
            filename (str): The name of the file to save the scaler to.

        Raises:
            ValueError: If the scaler is not fitted.

        Returns:
            None
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted")

        df = self.get_df()
        df.to_parquet(filename)

    def load(self, filename):
        """
        Load the scaler parameters from a file and fit the scaler.

        Args:
            filename (str): The path to the file containing the scaler parameters.

        Returns:
            None
        """
        self.df = pd.read_parquet(filename)
        self.range_dict = {feat: (min_x, max_x) for feat, min_x, max_x  in zip(self.df["feature_name"], self.df["min"], self.df["max"])}
        self.bit_shifts = {feat: bit_shift for feat, bit_shift in zip(self.df["feature_name"], self.df["bit_shift"])}
        self.target = (self.df["inf"][0], self.df["sup"][0])
        for key in self.range_dict:
            self.scale_funcs[key] = eval(
                f"lambda x: {self.target[0]}+(x-{self.range_dict[key][0]})/(2**{self.bit_shifts[key]})"
            )

        self.fitted = True

    def __str__(self):
        print("inf + (x - min) >> bit_shift")
        return self.get_df().__str__()

    def clear(self):
        return self.__init__()
