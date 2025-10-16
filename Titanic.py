import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = sns.load_dataset('titanic') 

print("Dataset shape:", df.shape)
print("Column types:\n", df.dtypes)
print("\nSummary statistics:\n", df.describe(include='all'))


df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()


sns.boxplot(data=df[['age', 'fare']])
plt.title('Boxplot of Age and Fare')
plt.show()


sns.pairplot(df[['age', 'fare', 'pclass', 'survived']].dropna(), hue='survived')
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['deck'].fillna('Unknown', inplace=True)


df.dropna(subset=['embarked'], inplace=True)


df_encoded = pd.get_dummies(df, columns=['sex', 'embarked', 'class', 'who', 'deck', 'embark_town', 'alive'], drop_first=True)


X = df_encoded[['pclass', 'age', 'fare', 'sex_male', 'embarked_Q', 'embarked_S']]
y = df['survived']



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
