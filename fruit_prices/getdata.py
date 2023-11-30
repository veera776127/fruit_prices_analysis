import pandas as pd

class FruitPricesDatasetLoader:
    """
    A class to load and retrieve the Fruit Prices 2020 dataset.

    This class provides methods to load the Fruit Prices 2020 dataset from a CSV file into a pandas DataFrame and retrieve it for analysis.

    Attributes
    ----------
    file_path : str
        The file path to the Fruit Prices 2020 dataset CSV file.
    fruit_prices_dataset : pandas.DataFrame or None
        The loaded Fruit Prices 2020 dataset, or None if the dataset hasn't been loaded yet.

    Methods
    -------
    load_fruit_prices_dataset():
        Loads the Fruit Prices 2020 dataset from the CSV file and stores it in fruit_prices_dataset attribute.
    get_fruit_prices_dataset():
        Returns the loaded Fruit Prices 2020 dataset if available, otherwise returns an error message.
    """

    def __init__(self, file_path):
        """
        Constructs all the necessary attributes for the FruitPricesDatasetLoader object.

        Parameters
        ----------
            file_path : str
                The file path to the Fruit Prices 2020 dataset CSV file.
        """
        self.file_path = file_path
        self.fruit_prices_dataset = None

    def load_fruit_prices_dataset(self):
        """
        Load the Fruit Prices 2020 dataset.

        Attempts to read the Fruit Prices 2020 dataset CSV file and store it in the fruit_prices_dataset attribute.
        Handles FileNotFoundError if the file is not found at the given path.

        Returns
        -------
        pandas.DataFrame or str
            The loaded Fruit Prices 2020 dataset DataFrame, or an error message if the file is not found.
        """
        try:
            self.fruit_prices_dataset = pd.read_csv(self.file_path)
            return self.fruit_prices_dataset
        except FileNotFoundError:
            return f"Error: File not found at {self.file_path}"

    def get_fruit_prices_dataset(self):
        """
        Retrieve the loaded Fruit Prices 2020 dataset.

        Returns the Fruit Prices 2020 dataset if it has been loaded. If the dataset is not loaded, returns an error message.

        Returns
        -------
        pandas.DataFrame or str
            The loaded Fruit Prices 2020 dataset DataFrame, or an error message if the dataset is not loaded.
        """
        if self.fruit_prices_dataset is not None:
            return self.fruit_prices_dataset
        else:
            return "Error: Fruit Prices 2020 dataset not loaded. Use load_fruit_prices_dataset() method first."
