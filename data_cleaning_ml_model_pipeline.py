
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset and memory optimize
dtypes = {'cc_num': 'int64', 'merchant': 'category', 'category': 'category',
          'amt': 'float32', 'is_fraud': 'int8', 'state': 'category'}
df = pd.read_csv('credit_card_transactions.csv', dtype=dtypes, parse_dates=['trans_date_trans_time'])

print(f'Shape: {df.shape}, Fraud rate:{df["is_fraud"].mean():.2%}')
# Handle missing values
df = df.dropna()
# Remove duplicates
df = df.drop_duplicates()
# Age column
df['dob'] = pd.to_datetime(df['dob'], format='%Y-%m-%d', errors='coerce')
df['age'] = 2026 - df['dob'].dt.year
# City pop
df['city_pop'] = pd.to_numeric(df['city_pop'], errors='coerce')
# Transaction hour
df['hour'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce').dt.hour

# Quick EDA plot
sample_df = df.sample(100000, random_state=42)

plt.figure(figsize=(10, 6))
sns.boxplot(x='is_fraud', y='amt', data=sample_df)
plt.yscale('log')
plt.title('Fraud vs Amount')
plt.savefig('fraud_amount.png', dpi=300)
plt.show()

print("Data cleaned df.shape:", df.shape)
df.to_csv('cleaned_transactions.csv', index=False)

# Fraud vs Category plot
plt.figure(figsize=(10, 6))
sns.countplot(x = 'category', hue = 'is_fraud', data = sample_df)
plt.yscale('log')
plt.xticks(rotation = 30, fontsize=8)
plt.title('Fraud vs Category')
plt.tight_layout()
plt.savefig('fraud_category.png', dpi=300)
plt.show()

# Age vs Fraud plot
sample_df = sample_df.dropna(subset=['age'])
plt.figure(figsize=(10, 6))
sns.boxplot(x = 'is_fraud', y ='age', data = sample_df )
plt.title('Fraud vs Age')
plt.savefig('fraud_age.png', dpi=300)
plt.show()

# Fraud Amount by Hour plot
plt.figure(figsize=(14,6))
sns.boxplot(x='hour', y='amt', hue='is_fraud', data=sample_df)
plt.yscale('log')  # Log scale - better visibility
plt.title('Fraud Amount by Hour (Log Scale) – Peak 22-3')
plt.xlabel('Transaction hours (24h)')
plt.ylabel('Amount (log $)')
plt.legend(title='Type')
plt.tight_layout()
plt.savefig('fraud_hour.png', dpi=300)
plt.show()

df_fraud = df[df['is_fraud']==1]
hour_fraud = df_fraud['hour'].value_counts().sort_index()
print('Fraud after hour:\n', hour_fraud.head(10))
print('Worst 5 hours:\n', hour_fraud.nlargest(5))

# Fraud RATE by hour
hour_stats = df.groupby('hour')['is_fraud'].agg(['count', 'mean']).round(4)
print('Fraud RATE by hour:\n', hour_stats.sort_values(by ='mean', ascending=False).head(10)) # mean *100 (%)

# Fraud RATE plot
hour_stats = hour_stats.reset_index()
hour_stats['mean_pct'] = hour_stats['mean'] * 100
hour_stats.plot(x='hour', y='mean_pct' , kind='bar', figsize=(12,6))
plt.title('Fraud RATE % by Hour')
plt.ylabel('Fraud %')
plt.xticks(rotation = 0, fontsize=6)
plt.axhline(y=hour_stats['mean'].mean()*100, color='red', linestyle='--', label='mean')
plt.axvspan(22, 24, alpha=0.2, color='red', label='night peak')
plt.axvspan(0, 3, alpha=0.2, color='red')
plt.legend()
plt.savefig('fraud_rate.png', dpi=300)
plt.show()


###### FRAUD DETECTION MODEL ######

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import  warnings
warnings.filterwarnings('ignore')

# 1. FEATURES
features = ['amt', 'hour', 'city_pop', 'age']

X = sample_df[features].fillna(sample_df[features].median()) # Fix NaN
y = sample_df['is_fraud']

# 2. TRAIN/TEST Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. SCALER
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Logistic Regression
lr_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# 5. Predictions
y_pred = lr_model.predict(X_test_scaled)
y_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# 6. Metrics
print(f'\nModel Performance: \n{classification_report(y_test, y_pred)}')
print(f'ROC AUC: \n{roc_auc_score(y_test, y_proba):.3f}')

print(f'Confusion Matrix (frauds detected):\n{confusion_matrix(y_test, y_pred)}')

# 7. FEATURE Importance
importances = pd.DataFrame(
    {
        'Feature': features,
        'Importance': lr_model.coef_[0]
    }
).sort_values(by='Importance', ascending=False, key=abs)
print(f'\nMost important Features:\n{importances.head(10)}')

# 8. ROC + Confusion Matrix visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ROC Curve
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_proba)
ax1.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_test, y_proba):.3f})')
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve')
ax1.legend()

# Confusion Matric heatmap
sns.heatmap(data=confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax2, cmap='Blues')
ax2.set_title('Confusion Matrix')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('fraud_model_results.png', dpi=300)
plt.show()