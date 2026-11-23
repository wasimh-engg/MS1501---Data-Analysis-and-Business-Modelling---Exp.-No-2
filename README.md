# Exp No.2 : Exploratory Data Analysis (EDA)

## Aim
To perform Exploratory Data Analysis (EDA) on the heart disease dataset to understand the data distribution, relationships, and key patterns useful for business insights and model building.

## Dataset
The dataset used for this experiment is `heartdisease.csv`. It contains medical and demographic information such as age, sex, cholesterol levels (`totChol`), `BMI`, blood pressure, and target values (`TenYearCHD`) indicating heart disease presence.

## Algorithm
1. Import required Python libraries (`numpy`, `pandas`, `seaborn`, `matplotlib`).
2. Load the dataset into a Pandas DataFrame.
3. Display the dataset structure, shape, and descriptive statistics.
4. Check for missing values and duplicate rows.
5. Examine correlations among numeric features using a correlation matrix and heatmap.
6. Visualize univariate distributions using histograms, boxplots, and countplots.
7. Analyze bivariate and multivariate relationships using scatter plots and pairplots.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Basic Exploration** | `df.info()` | Inspect dataset structure and data types |
| | `df.describe()` | Generate descriptive statistics |
| | `df.isnull().sum()` | Detect missing values in each column |
| | `df.duplicated().sum()` | Identify duplicate rows |
| | `df.corr()` | Find correlations between numeric variables |
| **Univariate Analysis** | `sns.histplot()` | Plot histogram (distribution of single variable) |
| | `sns.boxplot()` | Identify outliers using boxplot |
| | `sns.countplot()` | Count occurrences of categorical variables |
| **Bivariate Analysis** | `sns.scatterplot()` | Display relationship between two variables |
| | `sns.heatmap()` | Show correlation matrix visually |
| **Multivariate Analysis** | `sns.pairplot()` | Show pairwise relationships across variables |
| **Visualization Controls** | `plt.figure()`, `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.show()` | Configure and display plotted charts |

## Program & Output
# Experiment No. 2: Exploratory Data Analysis (EDA)

## Code

```python
# Exploratory Data Analysis (EDA) on the Heart Disease Dataset

# Step 1: Import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
sns.set_style("whitegrid")

# Step 2: Load the dataset
df = pd.read_csv("heartdisease.csv")

# Step 3: Structure, shape and descriptive statistics
print("Shape of dataset:", df.shape)
print("\nDataset Info:")
df.info()
print("\nDescriptive Statistics:")
print(df.describe())

# Step 4: Missing values and duplicates
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Step 5: Correlation matrix and heatmap
numeric_df = df.drop(columns=["S. No"]).select_dtypes(include=np.number)
corr = numeric_df.corr()
print("\nCorrelation of features with TenYearCHD:")
print(corr["TenYearCHD"].sort_values(ascending=False))

plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Heart Disease Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=120)
plt.show()

# Step 6: Univariate analysis
num_cols = ["Age", "BMI", "Totalcholestrol", "SysBP"]

# 6a. Histograms
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.ravel(), num_cols):
    sns.histplot(df[col], kde=True, bins=20, color="steelblue", ax=ax)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("images/histograms.png", dpi=120)
plt.show()

# 6b. Boxplots (outlier detection)
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.ravel(), num_cols):
    sns.boxplot(y=df[col], color="lightcoral", ax=ax)
    ax.set_title(f"Boxplot of {col}")
plt.tight_layout()
plt.savefig("images/boxplots.png", dpi=120)
plt.show()

# 6c. Countplots
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
sns.countplot(x="Gender", data=df, hue="Gender", palette="Set2", legend=False, ax=axes[0])
axes[0].set_title("Patients by Gender")
sns.countplot(x="currentsmoker", data=df, hue="currentsmoker", palette="Set1", legend=False, ax=axes[1])
axes[1].set_title("Current Smoker (0 = No, 1 = Yes)")
sns.countplot(x="TenYearCHD", data=df, hue="TenYearCHD", palette="Dark2", legend=False, ax=axes[2])
axes[2].set_title("TenYearCHD (0 = No, 1 = Yes)")
for ax in axes:
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("images/countplots.png", dpi=120)
plt.show()

print("\nTenYearCHD class distribution:")
print(df["TenYearCHD"].value_counts())
print(df["TenYearCHD"].value_counts(normalize=True).round(3))

# Step 7: Bivariate and multivariate analysis
# 7a. Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(x="SysBP", y="DiaBP", hue="TenYearCHD", data=df, palette="coolwarm", ax=axes[0])
axes[0].set_title("Systolic vs Diastolic Blood Pressure")
sns.scatterplot(x="Age", y="Totalcholestrol", hue="TenYearCHD", data=df, palette="coolwarm", ax=axes[1])
axes[1].set_title("Age vs Total Cholesterol")
plt.tight_layout()
plt.savefig("images/scatterplots.png", dpi=120)
plt.show()

# 7b. Pairplot
pair = sns.pairplot(df[["Age", "BMI", "SysBP", "Totalcholestrol", "TenYearCHD"]],
                    hue="TenYearCHD", palette="coolwarm", corner=True)
pair.figure.suptitle("Pairplot of Key Features", y=1.02, fontsize=14, fontweight="bold")
pair.savefig("images/pairplot.png", dpi=100)
plt.show()

Prevalentstroke    0.127883
Diabetes           0.126330
SysBP              0.114203
PrevalentHyp       0.052606
Age                0.034526
BPMeds             0.034323
DiaBP              0.028685
BMI                0.026675
Glucose           -0.015735
Heartrate         -0.044918
Totalcholestrol   -0.078204
currentsmoker     -0.082408
Education         -0.086376
Cigsperday        -0.096773

<img width="2752" height="1536" alt="img 1" src="https://github.com/user-attachments/assets/5f668f60-3d4f-47c2-bec3-904600540b2d" />
TenYearCHD class distribution:
0    204   (81.6%)
1     46   (18.4%)

## Result
Thus, Exploratory Data Analysis (EDA) on the heart disease dataset (heartdisease.csv) was executed successfully.
