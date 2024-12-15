import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
customers_df=pd.read_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Customers.csv')
d=pd.DataFrame(customers_df)
print(d)
subscriptions_df=pd.read_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Subscriptin (1).csv')
d=pd.DataFrame(subscriptions_df)
print(d)
transactions_df=pd.read_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Transcation (3).csv')
d=pd.DataFrame(transactions_df)
print(d)
Usage = pd.read_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Usage (1).csv')
d=pd.DataFrame(Usage)
print(d)
import pandas as pd

# Load CSV files
customers_df = pd.read_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Customers.csv')
subscriptions_df = pd.read_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Subscriptin (1).csv')
transactions_df = pd.read_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Transcation (3).csv')
usage_df = pd.read_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Usage (1).csv')

# Print DataFrames to confirm loading
print("Customers DataFrame:")
print(customers_df.head(), "\n")

print("Subscriptions DataFrame:")
print(subscriptions_df.head(), "\n")

print("Transactions DataFrame:")
print(transactions_df.head(), "\n")

print("Usage DataFrame:")
print(usage_df.head(), "\n")

# Merge DataFrames
merged_df = pd.merge(customers_df, subscriptions_df, on='CustomerID', how='outer')
merged_df = pd.merge(merged_df, transactions_df, on='CustomerID', how='outer')
merged_df = pd.merge(merged_df, usage_df, on='CustomerID', how='outer')

# Check for missing values
missing_values = merged_df.isnull().sum()
print("Missing Values per Column:")
print(missing_values)

# Merge tables into one consolidated DataFrame
consolidated_df = pd.merge(customers_df, subscriptions_df, on='CustomerID', how='outer')
consolidated_df = pd.merge(consolidated_df, transactions_df, on='CustomerID', how='outer')
consolidated_df = pd.merge(consolidated_df, usage_df, on='CustomerID', how='outer')

# Display the structure and a preview of the consolidated data
print("Consolidated DataFrame Overview:")
print(consolidated_df.info())
print(consolidated_df.head())

# Identify missing values
missing_summary = consolidated_df.isnull().sum()
print("Missing Values Summary:")
print(missing_summary)

 handling of missing values
# Drop rows where 'CustomerID' is missing
consolidated_df = consolidated_df.dropna(subset=['CustomerID'])

# Fill numerical missing values with mean
for col in consolidated_df.select_dtypes(include=['float64', 'int64']).columns:
    consolidated_df[col].fillna(consolidated_df[col].mean(), inplace=True)

# Fill categorical missing values with "Unknown"
for col in consolidated_df.select_dtypes(include=['object']).columns:
    consolidated_df[col].fillna("Unknown", inplace=True)

# Convert date columns to datetime
date_columns = ['StartDate', 'EndDate', 'transaction_date']  # Replace with actual date column names
for col in date_columns:
    consolidated_df[col] = pd.to_datetime(consolidated_df[col], errors='coerce')

# Check conversion
print("Date Columns Converted:")
print(consolidated_df[date_columns].dtypes)
# Calculate TenureDays 
consolidated_df['TenureDays'] = (consolidated_df['EndDate'] - consolidated_df['StartDate']).dt.days

# Handle cases where TenureDays might be 0 or negative
consolidated_df['TenureDays'] = consolidated_df['TenureDays'].apply(lambda x: x if x > 0 else None)

# Calculate AvgDailyUsage safely
consolidated_df['AvgDailyUsage'] = consolidated_df['amount'] / consolidated_df['TenureDays']

# Handle missing or invalid TenureDays by explicitly assigning the result
consolidated_df['AvgDailyUsage'] = consolidated_df['AvgDailyUsage'].fillna(0)

# Preview the result
print("Calculated AvgDailyUsage:")
print(consolidated_df[['CustomerID', 'amount', 'TenureDays', 'AvgDailyUsage']].head())

 # Handle cases where tenure might be NaN or zero
from datetime import datetime

# Define churn status
today = datetime.now()
consolidated_df['Churned'] = consolidated_df['EndDate'].apply(lambda x: x < today if pd.notnull(x) else False)

# Example of a usage metric calculation
# Calculate AvgDailyUsage
consolidated_df['AvgDailyUsage'] = consolidated_df['amount'] / consolidated_df['TenureDays']

# Handle missing or zero tenure cases explicitly
consolidated_df['AvgDailyUsage'] = consolidated_df['AvgDailyUsage'].fillna(0)


# Final structure and preview
print("Cleaned and Transformed DataFrame Overview:")
print(consolidated_df.info())
print(consolidated_df.head())

# Save the consolidated and cleaned data for future use
consolidated_df.to_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Consolidated_Cleaned_Data.csv', index=False)

#step3
import matplotlib.pyplot as plt
import seaborn as sns

# Age distribution
plt.figure(figsize=(6, 5))
sns.hist(consolidated_df['age'], kde=True, bins=20, color='purple')
plt.title('Customer Age Distribution')
plt.xlabel('age')
plt.ylabel('Frequency')
plt.show()

# Gender distribution
plt.figure(figsize=(4,4))
gender_counts = consolidated_df['Gender'].value_counts()
gender_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'pink'], explode=(0.1, 0), shadow=True)
plt.title('Gender Distribution')
plt.ylabel('')
plt.show()

# Income distribution
plt.figure(figsize=(8,5))
sns.boxplot(x=consolidated_df['Income'], color='red')
plt.title('Customer Income Distribution')
plt.xlabel('Income')
plt.show()


# Subscription status counts
subscription_status_counts = consolidated_df['Churned'].value_counts()

# Plot subscription status
plt.figure(figsize=(6, 5))
sns.barplot(x=subscription_status_counts.index, y=subscription_status_counts.values, palette='viridis')
plt.xticks(ticks=[0, 1], labels=['Active', 'Churned'])
plt.title('Subscription Status')
plt.xlabel('Status')
plt.ylabel('Number of Customers')
plt.show()



# Transaction type distribution
transaction_type_counts = consolidated_df['transaction_type'].value_counts()

# Plot transaction types
plt.figure(figsize=(8, 5))
sns.barplot(
   x=transaction_type_counts.index,
   y=transaction_type_counts.values,
    hue=transaction_type_counts.index,  # Assign x to hue
     palette='muted',
    dodge=False  # no separation for hue groups
  )
plt.title('Distribution of Transaction Types')
plt.xlabel('Transaction Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.legend([],[], frameon=False)  # Disable legend
plt.show()

from sklearn.preprocessing import LabelEncoder

# Create an encoder
encoder = LabelEncoder()

# Encode Churn column ('Yes' = 1, 'No' = 0)
consolidated_df['Churned'] = encoder.fit_transform(consolidated_df['Churned'])

# Encode categorical columns, e.g., Gender, Subscription Status
consolidated_df['Gender'] = encoder.fit_transform(consolidated_df['Gender'])
consolidated_df['Status'] = encoder.fit_transform(consolidated_df['Status'])

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
data = pd.read_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Customers.csv')
subscriptions_df=pd.read_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Subscriptin (1).csv')

# Label Encoding for categorical features
categorical_features = ['Gender', 'Income', 'region']
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le  # Store encoders for decoding later if needed

# Feature Engineering
data['Customer_Lifetime'] = (pd.to_datetime('today') - pd.to_datetime(data['Income'])).dt.days
['Average_Monthly_Spend'] = data['region'] / data['Months Subscribed']

# Scaling numerical features
scaler = StandardScaler()
data[['Customer_Lifetime', 'Average_Monthly_Spend']] = scaler.fit_transform(data[['Customer_Lifetime', 'Average_Monthly_Spend']])


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Define features and target
X = data.drop(columns=['CustomerID', 'Churn'])
y = data['Churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

