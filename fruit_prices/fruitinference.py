import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class Inference:
    """
    A class for performing various inference analyses on the 'Fruit Prices 2020' dataset.

    Methods
    -------
    correlation_analysis():
        Performs correlation analysis on the given data.

    linear_regression_analysis(x_column, y_column):
        Performs linear regression to predict one variable from another.

    scatter_plot(x_column, y_column):
        Creates a scatter plot of two specified columns.

    histogram(column):
        Creates a histogram of a specified column.

    box_plot(column):
        Creates a box plot of a specified column.
9    """

    def __init__(self, data):
        """
        Initializes the Inference object with the 'Fruit Prices 2020' dataset.

        Parameters:
        data (DataFrame): The dataset for analysis.
        """
        self.data = data

    def correlation_analysis(self):
        """
        Perform correlation analysis on the given data.

        Returns:
        None: Plots the correlation heatmap.
        """
        correlation_matrix = self.data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

    def linear_regression_analysis(self, x_column, y_column):
        """
        Perform linear regression to predict one variable from another.

        Parameters:
        x_column (str): The column name to use as predictor.
        y_column (str): The column name to use as response.

        Returns:
        dict: A dictionary containing regression results such as coefficients, mean squared error, and R-squared.
        """
        X = self.data[[x_column]]
        y = self.data[y_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        plt.scatter(X_test, y_test, color='blue')
        plt.plot(X_test, y_pred, color='red', linewidth=2)
        plt.title(f'Linear Regression of {y_column} on {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()

        return {'coefficients': model.coef_, 'mean_squared_error': mse, 'r_squared': r2}

    def scatter_plot(self, x_column, y_column):
        """
        Creates a scatter plot of two specified columns.

        Parameters:
        x_column (str): The column name for the x-axis.
        y_column (str): The column name for the y-axis.
        """
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.data[x_column], y=self.data[y_column])
        plt.title(f'Scatter Plot of {x_column} vs {y_column}')
        plt.show()

    def histogram(self, column):
        """
        Creates a histogram of a specified column.

        Parameters:
        column (str): The column name to plot the histogram for.
        """
        plt.figure(figsize=(8, 6))
        sns.histplot(self.data[column], kde=True)
        plt.title(f'Histogram of {column}')
        plt.show()

    def box_plot(self, column):
        """
        Creates a box plot of a specified column.

        Parameters:
        column (str): The column name to plot the box plot for.
        """
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=self.data[column])
        plt.title(f'Box Plot of {column}')
        plt.show()
