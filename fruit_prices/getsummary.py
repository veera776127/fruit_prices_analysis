import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataSummary:
    """
    A class for providing a comprehensive summary of a dataset.

    Thisimport pandas as pd

# Assuming the 'Fruit Prices 2020' dataset is loaded into a DataFrame named 'fruit_prices_df'
# If not, you can load it like this:
# fruit_prices_df = pd.read_csv('/path/to/Fruit Prices 2020.csv')

# Import the DataSummary class (make sure the class definition is included in the notebook or imported from a module)
from your_module import DataSummary  # Replace 'your_module' with the name of the module where DataSummary is defined if it's in a separate file

# Initialize the DataSummary object with the dataset
data_summary = DataSummary(fruit_prices_df)

# Display the first 5 rows of the dataset
data_summary.display_head()

# Display the shape of the dataset
print("Dataset Shape:", data_summary.display_shape())

# Display concise summary of the dataset
data_summary.display_info()

# Display descriptive statistics
print(data_summary.display_descriptive_statistics())

# Display missing values in each column
print("Missing Values in Each Column:\n", data_summary.display_missing_values())

# Display unique values in a specific column, for example, 'Fruit'
print("Unique Values in 'Fruit' Column:\n", data_summary.display_unique_values('Fruit'))

# Display data types of each column
print("Data Types of Each Column:\n", data_summary.display_data_types())

# Visualize the distribution of missing values in the dataset
data_summary.plot_missing_values()
 class includes various methods to display information about a dataset, aiding in understanding its structure, content, and potential issues before data cleaning, preprocessing, and exploratory data analysis.

    Attributes:
    ----------
    dataset : pandas.DataFrame
        The dataset to be summarized.

    Methods:
    -------
    display_head(n=5):
        Displays the first n rows of the dataset.

    display_shape():
        Shows the number of rows and columns in the dataset.

    display_info():
        Provides a concise summary of the dataset.

    display_descriptive_statistics():
        Shows descriptive statistics for numerical columns.

    display_missing_values():
        Indicates the number of missing values in each column.

    display_unique_values(column):
        Shows unique values in a specified column.

    display_data_types():
        Displays the data types of each column.

    plot_missing_values():
        Visualizes the distribution of missing values in the dataset.
    """

    def __init__(self, dataset):
        """
        Initializes the DataSummary with the specified dataset.
        """
        self.dataset = dataset

    def display_head(self, n=5):
        """
        Displays the first n rows of the dataset.
        """
        return self.dataset.head(n)

    def display_shape(self):
        """
        Displays the number of rows and columns in the dataset.
        """
        return self.dataset.shape

    def display_info(self):
        """
        Provides a concise summary of the dataset.
        """
        return self.dataset.info()

    def display_descriptive_statistics(self):
        """
        Shows descriptive statistics for numerical columns.
        """
        return self.dataset.describe()

    def display_missing_values(self):
        """
        Indicates the number of missing values in each column.
        """
        return self.dataset.isna().sum()

    def display_unique_values(self, column):
        """
        Shows unique values in a specified column.
        """
        return self.dataset[column].unique()

    def display_data_types(self):
        """
        Displays the data types of each column.
        """
        return self.dataset.dtypes

    def plot_missing_values(self):
        """
        Visualizes the distribution of missing values in the dataset.
        """
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.dataset.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title("Distribution of Missing Values")
        plt.show()
