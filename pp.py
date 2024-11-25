import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
Customers=pd.read_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Customers.csv')
d=pd.DataFrame(Customers)
print(d)
Subscriptions=pd.read_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Subscriptin (1).csv')
d=pd.DataFrame(Subscriptions)
print(d)
Transactions=pd.read_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Transcation (3).csv')
d=pd.DataFrame(Transactions)
print(d)
Usage = pd.read_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Usage (1).csv')
d=pd.DataFrame(Usage)
print(d)
merged_df = pd.merge(Customers, Subscriptions, on="CustomerID", how="left")
print(merged_df)
merged_df = merged_df.merge(Transactions, on="CustomerID", how="left")
print(merged_df)
merged_df = merged_df.merge(Usage, on="CustomerID", how="left")     
print(merged_df)
print(merged_df[["StartDate", "EndDate"]].dtypes)
print(merged_df[merged_df["StartDate"].isna() | merged_df["EndDate"].isna()])
print(merged_df.columns)
merged_df["StartDate"] = pd.to_datetime(merged_df["StartDate"], errors="coerce")
print(merged_df["StartDate"])
merged_df["EndDate"] = pd.to_datetime(merged_df["EndDate"], errors="coerce")
print(merged_df["EndDate"])
merged_df["tenure"] = (merged_df["EndDate"] - merged_df["StartDate"]).dt.days
print(merged_df["tenure"])
print(merged_df[["StartDate", "EndDate", "tenure"]].head())
merged_df["tenure"] = merged_df["tenure"].replace(0, np.nan)
print(merged_df["tenure"])
merged_df["average_monthly_spend"] = merged_df["amount"] / (merged_df["tenure"] / 30.0)
print(merged_df["average_monthly_spend"])
merged_df["average_monthly_spend"].fillna(0, inplace=True)
print(merged_df.columns)
print(merged_df.head())
plt.figure(figsize=(2, 3))
sns.histplot(merged_df["Age"], bins=10, kde=True)
plt.title("Age Distribution")
plt.show()

Subscriptions = pd.read_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Subscriptin (1).csv')
print(Subscriptions.columns)
plt.figure(figsize=(5, 5))
sns.countplot(x="Status", data=Subscriptions)
plt.title("Subscriptions Status")
plt.show()
print(Subscriptions.columns)
plt.figure(figsize=(4, 5))
sns.countplot(x="transaction_type", data=Transactions)
plt.title("Transaction Types")
plt.show()
















