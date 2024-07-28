# Car-Price-Prediction-Model

**Overview**
This project involves building a car price prediction model using a dataset containing various car attributes. The model employs two regression techniques: Linear Regression and Lasso Regression. The goal is to predict the selling price of cars based on features such as fuel type, seller type, transmission, and ownership history.

**Dependencies**
The following libraries are used in this project:

pandas: For data manipulation and analysis
matplotlib: For data visualization
seaborn: For enhanced data visualization
sklearn: For model building and evaluation
Data Collection and Processing
Loading the Dataset
The dataset is loaded into a pandas DataFrame from a CSV file.

python
Copy code
car_dataset = pd.read_csv('/content/CAR DETAILS FROM CAR DEKHO.csv')
Data Exploration
First Five Rows: car_dataset.head()
Shape of Dataset: car_dataset.shape
Dataset Information: car_dataset.info()
Missing Values: car_dataset.isnull().sum()
Categorical Data Distribution:
python
Copy code
print(car_dataset.fuel.value_counts())
print(car_dataset.seller_type.value_counts())
print(car_dataset.transmission.value_counts())
print(car_dataset.owner.value_counts())
**Data Encoding**
Categorical variables are encoded into numerical values:

Fuel Type: Petrol (0), Diesel (1), CNG (2), LPG (3), Electric (4)
Seller Type: Dealer (0), Individual (1), Trustmark Dealer (2)
Transmission: Manual (0), Automatic (1)
Owner: First Owner (0), Second Owner (1), Third Owner (2), Fourth & Above Owner (3), Test Drive Car (4)
python
Copy code
car_dataset.replace({'fuel':{'Petrol':0,'Diesel':1,'CNG':2,'LPG':3,'Electric':4}}, inplace=True)
car_dataset.replace({'seller_type':{'Dealer':0,'Individual':1,'Trustmark Dealer':2}}, inplace=True)
car_dataset.replace({'transmission':{'Manual':0,'Automatic':1}}, inplace=True)
car_dataset.replace({'owner':{'First Owner':0,'Second Owner':1,'Third Owner':2,'Fourth & Above Owner':3,'Test Drive Car':4}}, inplace=True)
**Splitting the Data**
The data is split into features (X) and target variable (Y). The target variable is the selling price.

python
Copy code
X = car_dataset.drop(['name','selling_price'], axis=1)
Y = car_dataset['selling_price']
**To handle any non-numeric data:**

python
Copy code
X = X.apply(pd.to_numeric, errors='coerce')
Y = pd.to_numeric(Y, errors='coerce')
X = X.dropna()
Y = Y[X.index]
**Model Training**
1. Linear Regression
A Linear Regression model is trained using the training dataset.

python
Copy code
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)
**Model Evaluation**
**Training Data Prediction:**
python
Copy code
training_data_prediction = lin_reg_model.predict(X_train)
R Squared Error:
python
Copy code
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R Squared Error:", error_score)
Visualize actual vs. predicted prices:

python
Copy code
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Price")
plt.show()
**Test Data Prediction:**
python
Copy code
test_data_prediction = lin_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R Squared Error:", error_score)
2. Lasso Regression
A Lasso Regression model is trained similarly.

python
Copy code
lass_reg_model = Lasso()
lass_reg_model.fit(X_train, Y_train)
Model Evaluation
**Training Data Prediction:**

python
Copy code
training_data_prediction = lass_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R Squared Error:", error_score)
**Test Data Prediction:**

python
Copy code
test_data_prediction = lass_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R Squared Error:", error_score)
Visualizations are also provided for the Lasso Regression model.

**Conclusion**
This project demonstrates how to use Linear and Lasso Regression to predict car prices based on various features. The dataset was processed, encoded, and split into training and test sets. Both models were evaluated using R Squared Error, and visualizations of the actual versus predicted prices were provided.
