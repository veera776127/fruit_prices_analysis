import pandas as pd

class FruitPricesDataCleaning:
    """
    A class for performing data cleaning tasks on the 'Fruit Prices 2020' dataset.

    This class provides methods for handling missing values, correcting data types, and other cleaning tasks specific to the 'Fruit Prices 2020' dataset.

    Attributes:
    ----------
    dataset : pandas.DataFrame
        The 'Fruit Prices 2020' dataset to be cleaned.

    Methods:
    -------
    handle_missing_values(strategy='drop', columns=None):
        Handles missing values in the dataset based on the specified strategy.

    correct_data_types(column_type_dict):
        Corrects the data types of specified columns based on the provided dictionary.

    custom_cleaning():
        Performs any additional custom data cleaning tasks specific to the dataset.
    """

    def __init__(self, dataset):
        """
        Initializes the FruitPricesDataCleaning with the specified 'Fruit Prices 2020' dataset.
        """
        self.dataset = dataset

    def handle_missing_values(self, strategy='drop', columns=None):
        """
        Handles missing values in the dataset.

        Parameters:
        ----------
        strategy : str, optional
            The strategy to handle missing values, 'drop' to drop rows/columns, 'fill' to fill with a value (default is 'drop').
        columns : list, optional
            The list of columns to apply the missing value strategy (default is None, applying to all columns).
        """
        if strategy == 'drop':
            if columns:
                self.dataset.dropna(subset=columns, inplace=True)
            else:
                self.dataset.dropna(inplace=True)
        elif strategy == 'fill':
            fill_value = input("Enter the fill value: ")
            if columns:
                self.dataset[columns].fillna(fill_value, inplace=True)
            else:
                self.dataset.fillna(fill_value, inplace=True)

    def correct_data_types(self, column_type_dict):
        """
        Corrects the data types of specified columns.

        Parameters:
        ----------
        column_type_dict : dict
            A dictionary mapping column names to their correct data types.
        """
        for column, dtype in column_type_dict.items():
            self.dataset[column] = self.dataset[column].astype(dtype)


