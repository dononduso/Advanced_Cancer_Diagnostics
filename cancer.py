!pip3 install -U ucimlrepo  pandas matplotlib seaborn scikit-learn

import pandas as pd
from sklearn.datasets import load_breast_cancer

# 1. Load the dataset
data = load_breast_cancer()
cancer_df = pd.DataFrame(data.data, columns=data.feature_names)

# 2. Add the target (Diagnosis)
cancer_df['target'] = data.target 
# 0 = Malignant, 1 = Benign

print("Advanced Dataset Loaded!")
print(f"Features: {cancer_df.shape[1]}")
print(cancer_df.head())



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Scale the data
features = cancer_df.drop('target', axis=1)
scaled_features = StandardScaler().fit_transform(features)

# 2. Run PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(scaled_features)

# 3. Plotting
plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_results[:,0], y=pca_results[:,1], hue=cancer_df['target'], palette='viridis')
plt.title('PCA of Breast Cancer Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()



# 1. Verify data types and nulls
print(cancer_df.info())

# 2. Check the Target balance
print("--- Target Counts (0=M, 1=B) ---")
print(cancer_df['target'].value_counts(normalize=True))

# 3. Describe the spread of the first 5 features
print(cancer_df.iloc[:, 0:5].describe())


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Separate features from target
X = cancer_df.drop('target', axis=1)
y = cancer_df['target']

# 2. Standardize the data (CRITICAL for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Create a DataFrame for visualization
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Diagnosis'] = y.map({0: 'Malignant', 1: 'Benign'})

# 5. Visualize the Clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Diagnosis', palette='coolwarm', alpha=0.8)
plt.title('PCA Risk Map: 30D reduced to 2D')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Split the data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize and Train the Random Forest
# n_estimators=100 means we are using 100 individual decision trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 3. Evaluate the Model
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))

# 4. Feature Importance
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh', color='teal')
plt.title('Top 10 Diagnostic Predictors')
plt.show()

