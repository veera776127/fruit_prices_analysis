import pandas as pd
from fruit_prices.getdata import FruitPricesDatasetLoader
from fruit_prices.getsummary import DataSummary
from fruit_prices.dataclean import FruitPricesDataCleaning
from fruit_prices.fruitinference import Inference
from fruit_prices.fruitprediction import FruitPricePrediction
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "https://raw.githubusercontent.com/veera776127/project3/main/Datasets/Fruit%20Prices%202020.csv"
fruit_prices_dataset_loader = FruitPricesDatasetLoader(file_path)
loaded_dataset = fruit_prices_dataset_loader.load_fruit_prices_dataset()
retrieved_dataset = fruit_prices_dataset_loader.get_fruit_prices_dataset()

# Display descriptive statistics
data_summary = DataSummary(retrieved_dataset)
#print(data_summary.display_descriptive_statistics())

# Data cleaning
cleaner = FruitPricesDataCleaning(retrieved_dataset)
cleaner.handle_missing_values(strategy='drop')
column_types = {'RetailPrice': 'float', 'Yield': 'float'}
cleaner.correct_data_types(column_types)
cleaned_dataset = cleaner.dataset

# Prepare data for modeling
fruit_prices_df = cleaned_dataset

# Defining categorical and numerical columns, target column
categorical_cols = ['Form', 'RetailPriceUnit', 'CupEquivalentUnit']
numerical_cols = ['RetailPrice', 'Yield', 'CupEquivalentSize', 'CupEquivalentPrice']
target_col = 'RetailPrice'

# Creating an instance of FruitPricePrediction class
model = FruitPricePrediction(categorical_cols, numerical_cols, target_col)

# Preprocess the data
encoded_df = model.preprocess_data(fruit_prices_df)

# Split the data into features (X) and target variable (y)
X = encoded_df.drop(target_col, axis=1)
y = fruit_prices_df[target_col]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection
selected_features, X_train_selected = model.feature_selection(X_train, y_train)
X_test_selected = model.feat_selector.transform(X_test)
print("--------------------PREDICTING RETAIL PRICE BASED ON YIELD---------------------")

# Perform linear regression analysis
inference = Inference(cleaned_dataset)
results = inference.linear_regression_analysis('Yield', 'RetailPrice')

# Print analysis results
print("Coefficients:", results['coefficients'])
print("R-squared:", results['r_squared'])
print("Mean Squared Error:", results['mean_squared_error'])

print("\n\n--------------PREDICTING RETAIL PRICE USING ALL OTHER FEATURES-----------------")

# Train the model
print("Please wait for results...Training the model in progress.. \n")
model.train_model(X_train_selected, y_train)

print("Model training done.. Printing evaluation results \n")

# Evaluate the model
r2, mse = model.evaluate_model(X_test_selected, y_test)
print(f"R-squared (R2): {r2}")
print(f"Mean Squared Error (MSE): {mse}")

# Printing the model accuracy
model.print_training_accuracy(X_train_selected, y_train)

# Plotting actual vs. predicted values
model.plot_predictions(y_test, model.best_rf.predict(X_test_selected))





