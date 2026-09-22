# Diabetes Patient Data Preprocessing

## 📌 Project Overview

This project performs an end-to-end **data preprocessing and transformation workflow** on a diabetes patient dataset collected from healthcare-related sources.

The objective is to clean, standardize, analyze, and transform the raw dataset so that it is suitable for subsequent **exploratory data analysis and machine learning model development**.

The project demonstrates practical use of **Python, Pandas, NumPy, Matplotlib, and Scikit-learn** for handling real-world data quality issues.

---

## 🎯 Objectives

The main objectives of this project are to:

* Load and inspect the diabetes dataset
* Identify numerical and categorical variables
* Rename unclear column names
* Examine categorical values
* Handle missing values
* Identify and remove duplicate records
* Generate statistical summaries
* Visualize numerical distributions using box plots
* Handle outliers using specified statistical techniques
* Encode categorical variables
* Standardize numerical features
* Prepare the dataset for machine learning

---

## 📊 Dataset

**Dataset:** Diabetes Patient Dataset

**Source:**
[Diabetes Dataset – GitHub](https://raw.githubusercontent.com/GeethaGunasekaran1/Dataset_rep/refs/heads/main/diabetes.csv)

The original dataset contains **1,009 records and 15 columns**.

### Important Variables

| Column      | Description                  |
| ----------- | ---------------------------- |
| `ID`        | Original visit identifier    |
| `No_Pation` | Original patient identifier  |
| `Gender`    | Patient gender               |
| `AGE`       | Patient age                  |
| `Urea`      | Urea measurement             |
| `Cr`        | Creatinine measurement       |
| `HbA1c`     | HbA1c measurement            |
| `Chol`      | Cholesterol                  |
| `TG`        | Triglycerides                |
| `HDL`       | High-density lipoprotein     |
| `LDL`       | Low-density lipoprotein      |
| `VLDL`      | Very-low-density lipoprotein |
| `BMI`       | Body Mass Index              |
| `CLASS`     | Diabetes classification      |

### Diabetes Classification

| Value | Meaning      |
| ----- | ------------ |
| `N`   | No Diabetes  |
| `P`   | Pre-Diabetes |
| `Y`   | Diabetes     |

---

# 🔍 Project Workflow

The project is divided into four major stages.

## Task 1 – Data Loading & Inspection

The dataset was loaded using Pandas and inspected using:

```python
df.head()
df.tail()
df.shape
df.columns
df.info()
df.dtypes
```

Numerical and categorical columns were identified using Pandas `select_dtypes()`.

The initial inspection identified patient identifiers, demographic information, clinical measurements, and diabetes classification variables.

---

## Task 2 – Data Cleaning

### Column Renaming

The following columns were renamed to improve clarity:

```text
ID          → Visit_ID
No_Pation   → Patient_ID
```

The automatically generated `Unnamed: 0` index column was removed.

### Categorical Data Validation

The following categories were checked:

**Gender**

```text
F = Female
M = Male
```

**CLASS**

```text
N = No Diabetes
P = Pre-Diabetes
Y = Diabetes
```

Categorical values were standardized by removing unnecessary spaces and converting values to uppercase.

### Statistical Analysis

Statistical summaries were generated using:

* Mean
* Median
* Minimum
* Maximum
* Standard deviation

Clinical numerical variables were analyzed to understand their distributions and identify potentially extreme observations.

### Missing Values

Missing values were identified using:

```python
df.isnull().sum()
```

Imputation strategies were selected according to variable type:

* **Numerical variables:** Median
* **Categorical variables:** Mode

Median was selected for numerical variables because it is less affected by extreme observations.

### Duplicate Records

Duplicate rows were identified using:

```python
df.duplicated()
```

and removed using:

```python
df.drop_duplicates()
```

---

# 📦 Task 3 – Outlier Handling

Different outlier-handling strategies were applied according to the assignment requirements.

### Outliers Retained

Outliers in the following variables were deliberately retained:

* `AGE`
* `HbA1c`
* `BMI`

These observations may represent legitimate patient characteristics or clinically meaningful extreme values.

### Creatinine (`Cr`)

Values exceeding the **99.5th percentile** were removed.

```python
cr_threshold = df_clean['Cr'].quantile(0.995)

df_clean = df_clean[
    df_clean['Cr'] <= cr_threshold
]
```

### Urea

Values exceeding the **99.9th percentile** were removed.

```python
urea_threshold = df_clean['Urea'].quantile(0.999)

df_clean = df_clean[
    df_clean['Urea'] <= urea_threshold
]
```

The assignment-specific percentile thresholds were used instead of applying IQR to these two variables.

### Lipid Variables

Extreme observations in the following lipid-related variables were handled using the **IQR method**:

* `LDL`
* `VLDL`
* `HDL`
* `TG`
* `Chol`

The standard IQR boundaries were calculated as:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Observations outside these boundaries were removed.

---

# 🔢 Task 4 – Data Transformation

## Gender Encoding

The categorical `Gender` variable was transformed using **One-Hot Encoding**.

One-Hot Encoding was selected because Gender is a nominal categorical variable without a meaningful numerical order.

Scikit-learn's `OneHotEncoder` was used for this transformation.

Example:

```text
Gender
   ↓
One-Hot Encoding
   ↓
Gender_M
```

For a binary variable, one category can be dropped to avoid redundant dummy variables.

---

## Feature Scaling

The following clinical numerical features were standardized:

```text
AGE
Urea
Cr
HbA1c
Chol
TG
HDL
LDL
VLDL
BMI
```

`StandardScaler` from Scikit-learn was used.

Standardization transforms features so that they are centered around approximately zero with unit variance.

### Why Standardization?

The variables have different units and numerical ranges. Standardization places them on a comparable scale and is useful for machine-learning algorithms that are sensitive to feature magnitude.

### Variables Not Scaled

The following were excluded from feature scaling:

* `Visit_ID`
* `Patient_ID`
* `CLASS`

`Visit_ID` and `Patient_ID` are identifiers rather than meaningful measurements.

`CLASS` is the target variable and should be handled separately during model development.

---

# 🛠️ Technologies Used

| Technology                      | Purpose                             |
| ------------------------------- | ----------------------------------- |
| Python                          | Programming language                |
| Pandas                          | Data loading, cleaning and analysis |
| NumPy                           | Numerical operations                |
| Matplotlib                      | Data visualization                  |
| Scikit-learn                    | Encoding and feature scaling        |
| Google Colab / Jupyter Notebook | Development environment             |
| GitHub                          | Project and dataset repository      |

---

# 📁 Suggested Project Structure

```text
Diabetes-Data-Preprocessing/
│
├── README.md
│
├── diabetes.csv
│
├── Diabetes_Data_Preprocessing.ipynb
│
└── outputs/
    ├── statistical_summary.csv
    └── boxplots.png
```

---

# 📈 Key Findings

The preprocessing analysis identified several data-quality considerations in the raw dataset.

### Data Quality

* The dataset contained a mixture of numerical and categorical variables.
* Column names required clarification.
* Categorical values were reviewed for consistency.
* Missing observations were handled using appropriate imputation strategies.
* Duplicate records were identified and removed.

### Outlier Analysis

* `AGE`, `HbA1c`, and `BMI` outliers were retained because extreme values may have clinical significance.
* `Cr` values above the 99.5th percentile were removed.
* `Urea` values above the 99.9th percentile were removed.
* Extreme observations in lipid-related variables were handled using the IQR method.

### Data Transformation

* `Gender` was converted into numerical form using One-Hot Encoding.
* Clinical numerical variables were standardized using `StandardScaler`.
* Patient and visit identifiers were excluded from scaling.
* `CLASS` was preserved as the target variable.

---

# ✅ Final Outcome

After completing the preprocessing workflow, the dataset was transformed from a raw healthcare dataset into a cleaner and more consistent dataset suitable for subsequent machine-learning analysis.

The preprocessing pipeline includes:

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Column Standardization
     ↓
Categorical Validation
     ↓
Missing Value Treatment
     ↓
Duplicate Removal
     ↓
Statistical Analysis
     ↓
Outlier Handling
     ↓
Categorical Encoding
     ↓
Feature Scaling
     ↓
Machine-Learning Ready Dataset
```

---

# 🚀 Future Work

The processed dataset can be used for further machine-learning tasks such as:

* Exploratory Data Analysis
* Feature Selection
* Train-Test Split
* Classification Modeling
* Logistic Regression
* Decision Trees
* Random Forest
* Support Vector Machines
* K-Nearest Neighbors
* Model Evaluation
* Confusion Matrix Analysis
* Precision, Recall and F1-Score comparison

---

# 📚 References

* [Pandas Documentation](https://pandas.pydata.org/docs/)
* [NumPy Documentation](https://numpy.org/doc/)
* [Matplotlib Documentation](https://matplotlib.org/stable/)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [GitHub Documentation](https://docs.github.com/)

---

## 👤 Project Information

**Project:** Diabetes Patient Data Preprocessing
**Domain:** Healthcare Analytics / Data Science
**Tools:** Python, Pandas, NumPy, Matplotlib, Scikit-learn
**Environment:** Google Colab / Jupyter Notebook

---

This project focuses on **data preprocessing and preparation**. It does not provide medical diagnoses or clinical recommendations. The processed data is intended for educational data-science and machine-learning purposes.
