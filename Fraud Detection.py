# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.layers import Dense, Dropout
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score


# In[2]:


# Load the dataset
df = pd.read_csv("PS_20174392719_1491204439457_log.csv")


# In[3]:


# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())


# In[4]:


# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())


# In[5]:


# Exploratory Data Analysis (EDA)
print("\nDataset Summary:")
print(df.describe())


# In[6]:


# Visualizing the distribution of transactions types
plt.figure(figsize=(8, 6))
sns.countplot(x='type', data=df)
plt.title('Transaction Types Distribution')
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.show()


# In[7]:


# Visualizing fraud vs non-fraud transactions
plt.figure(figsize=(6, 6))
sns.countplot(x='isFraud', data=df)
plt.title('Fraudulent vs Non-Fraudulent Transactions')
plt.xlabel('Fraudulent')
plt.ylabel('Count')
plt.show()


# In[8]:


# Boxplot for transaction amounts by type
plt.figure(figsize=(10, 6))
sns.boxplot(x='type', y='amount', data=df)
plt.title('Transaction Amounts by Type')
plt.xlabel('Transaction Type')
plt.ylabel('Transaction Amount')
plt.yscale('log')
plt.show()


# In[13]:


# Box plot to analyze the amounts for fraudulent and non-fraudulent transactions
plt.figure(figsize=(10, 6))
sns.boxplot(x="isFraud", y="amount", data=df, showfliers=False, palette="Set2")
plt.title("Transaction Amounts: Fraudulent vs Non-Fraudulent")
plt.xlabel("Is Fraud")
plt.ylabel("Transaction Amount")
plt.xticks([0, 1], ["Non-Fraudulent", "Fraudulent"])
plt.show()


# In[14]:


# Distribution of transaction amounts
plt.figure(figsize=(10, 6))
sns.histplot(df["amount"], bins=50, kde=True, color="blue")
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.show()


# In[15]:


# Encode categorical features
df['type'] = LabelEncoder().fit_transform(df['type'])


# In[16]:


# Feature Engineering: Create new features
df['errorOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
df['errorDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']


# In[17]:


# Drop irrelevant columns
df = df.drop(['nameOrig', 'nameDest'], axis=1)


# In[18]:


# Preprocessing: Define features and target
X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df['isFraud']


# In[19]:


# Handle class imbalance using upsampling
df_majority = df[df.isFraud == 0]
df_minority = df[df.isFraud == 1]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)

df_upsampled = pd.concat([df_majority, df_minority_upsampled])
X = df_upsampled.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df_upsampled['isFraud']


# In[20]:


# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[21]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[23]:


# Model Training and Evaluation
# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)
y_prob_logistic = logistic_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Evaluation:")
print(classification_report(y_test, y_pred_logistic))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_logistic):.4f}")


# In[24]:


# Plot ROC curve for Logistic Regression
fpr, tpr, _ = roc_curve(y_test, y_prob_logistic)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="Logistic Regression (AUC = {:.4f})".format(roc_auc_score(y_test, y_prob_logistic)))
plt.title('ROC Curve - Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[32]:


# Neural Network Model
nn_model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

nn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
nn_model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2, verbose=1)


# In[33]:


# Neural Network Evaluation
nn_predictions = (nn_model.predict(X_test) > 0.5).astype("int32")
print("Performance of Neural Network:")
print("Accuracy:", accuracy_score(y_test, nn_predictions))


# In[35]:


# Predict probabilities for AUC-ROC calculation
nn_probabilities = nn_model.predict(X_test)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, nn_probabilities)
auc_score = roc_auc_score(y_test, nn_probabilities)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Neural Network (AUC = {auc_score:.4f})", color="darkorange")
plt.plot([0, 1], [0, 1], "r--")
plt.title("ROC Curve - Neural Network")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


# In[38]:


# Calculate ROC for Logistic Regression
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_prob_logistic)
auc_logistic = roc_auc_score(y_test, y_prob_logistic)

# Calculate ROC for Neural Network
fpr_nn, tpr_nn, _ = roc_curve(y_test, nn_probabilities)
auc_nn = roc_auc_score(y_test, nn_probabilities)


# In[43]:


# Plot the comparison
plt.figure(figsize=(8, 6))
plt.plot(fpr_logistic, tpr_logistic, label=f"Logistic Regression (AUC = {auc_logistic:.4f})", color="blue")
plt.plot(fpr_nn, tpr_nn, label=f"Neural Network (AUC = {auc_nn:.4f})", color="darkorange")

# Adding plot details
plt.title("ROC Curve Comparison: Logistic Regression vs Neural Network")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


# In[ ]:




