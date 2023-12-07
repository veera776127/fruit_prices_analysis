import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import time


class FruitPricePrediction:
    """
    FruitPricePrediction: A class to predict fruit prices based on given features.

    Attributes:
    - categorical_cols (list): List of column names containing categorical data.
    - numerical_cols (list): List of column names containing numerical data.
    - target_col (str): Name of the target column.
    - preprocessor (ColumnTransformer): Preprocessing steps for data transformation.
    - feat_selector (SelectFromModel): Feature selector using RandomForestRegressor.
    - best_rf (RandomForestRegressor): Best RandomForestRegressor model after training.
    """

    def __init__(self, categorical_cols, numerical_cols, target_col):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.target_col = target_col
        self.preprocessor = None
        self.feat_selector = None
        self.best_rf = None

    def preprocess_data(self, data):
        """
        Preprocesses the data by imputing missing values and encoding categorical features.

        Args:
        - data (DataFrame): Input dataset to be preprocessed.

        Returns:
        - encoded_df (DataFrame): Processed DataFrame after imputation and encoding.
        """
        numerical_transformer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )

        encoded_data = self.preprocessor.fit_transform(data)
        columns_after_encoding = list(self.numerical_cols) + list(
            self.preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(
                self.categorical_cols))
        encoded_df = pd.DataFrame(encoded_data, columns=columns_after_encoding)
        return encoded_df

    def feature_selection(self, X, y):
        """
        Performs feature selection using RandomForestRegressor.

        Args:
        - X (DataFrame): Input features.
        - y (Series): Target variable.

        Returns:
        - selected_features (array): Selected features after feature selection.
        - transformed_X (array): Transformed feature set.
        """
        self.feat_selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
        self.feat_selector.fit(X, y)
        selected_features = X.columns[self.feat_selector.get_support()]
        return selected_features, self.feat_selector.transform(X)

    def train_model(self, X_train, y_train):
        """
        Trains the RandomForestRegressor model using GridSearchCV.

        This method performs model training using RandomForestRegressor
        with hyperparameter tuning via GridSearchCV.

        Args:
        - X_train (DataFrame): Training features.
        - y_train (Series): Training target variable.

        Returns:
        - total_time (float): Total time taken for model training.

        Usage:
        model = FruitPricePrediction()
        total_training_time = model.train_model(X_train, y_train)
        """

        start_time = time.time()  # Record start time

        # Define hyperparameter grid for RandomForestRegressor
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30]
        }

        # Initialize base RandomForestRegressor
        rf = RandomForestRegressor(random_state=42)

        # Setup GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

        # Initialize the best RandomForestRegressor model with optimized hyperparameters
        self.best_rf = RandomForestRegressor(**best_params, random_state=42)
        self.best_rf.fit(X_train, y_train)

        end_time = time.time()  # Record end time
        total_time = end_time - start_time  # Calculate total time

        print(f"Time taken for model training: {total_time:.4f} seconds")

        return total_time  # Return total time taken for training

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the model using test data.

        Args:
        - X_test (DataFrame): Test features.
        - y_test (Series): Test target variable.

        Returns:
        - r2 (float): R-squared score.
        - mse (float): Mean squared error.
        """
        predictions = self.best_rf.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        return r2, mse

    def print_training_accuracy(self, X_train_selected, y_train):
        """
        Prints the training accuracy of the model.

        Args:
        - X_train_selected (DataFrame): Selected training features.
        - y_train (Series): Training target variable.
        """
        self.best_rf.fit(X_train_selected, y_train)
        training_accuracy = self.best_rf.score(X_train_selected, y_train)
        print(f"Model Training Accuracy: {training_accuracy}")

    def plot_predictions(self, y_test, predictions):
        """
        Plots the actual vs. predicted values.

        Args:
        - y_test (Series): Actual target values.
        - predictions (array): Predicted values.
        """
        plt.figure(figsize=(8, 6))
        
        # Plotting the actual values with a label
        plt.scatter(y_test, predictions, color='blue', alpha=0.5, label='Actual values')
        
        # Plotting the regression line with a label
        sns.regplot(x=y_test, y=predictions, scatter=False, line_kws={'color': 'red'}, label='Regression Line')
        
        plt.xlabel('Actual Values (RetailPrice)')
        plt.ylabel('Predicted Values (RetailPrice)')
        plt.title('Actual vs. Predicted Values')
        plt.legend()  # Add legend for both actual points and regression line
        plt.show()

