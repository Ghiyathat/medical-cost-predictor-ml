# medical-cost-predictor-ml
## **Table of Contents**

1. [Project Executive Summary](#1-project-executive-summary)
2. [The Dataset](#2-the-dataset)
3. [The Analytics Stack](#3-the-analytics-stack)
4. [Data Wrangling & Feature Engineering](#4-data-wrangling--feature-engineering)
5. [Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
6. [Machine Learning Strategy](#6-machine-learning-strategy)
7. [Key Outcomes & Model Performance](#7-key-outcomes--model-performance)
8. [Challenges & Data Limitations](#8-challenges--data-limitations)

### **1. Project Executive Summary**

Managing healthcare expenditures is a significant financial challenge for both individuals and insurance providers. The goal of this project was to develop a high-precision predictive model to forecast **annual medical costs** based on a patient’s demographic profile, lifestyle habits, and clinical history.

By analyzing a dataset of **$10,000$ patient records**, I moved beyond simple averages to create a system that understands how intersecting factors—such as smoking, high BMI, and chronic conditions—compounds health spending.

**Core Accomplishments:**

* **Predictive Accuracy:** Utilizing an ensemble **Random Forest Regressor**, the model achieved an **$R^{2}$ score of $0.9789$**, meaning it can explain approximately **$98\%$** of the variance in medical costs.
* **Error Minimization:** The model reached a **Mean Absolute Error (MAE) of $\$633.12$**, ensuring that predictions remain within a highly reliable range for financial planning.
* **Strategic Insights:** Beyond just prediction, the analysis identifies exactly which health triggers (like `previous_year_cost` and `medication_count`) are the primary drivers of future spending.

### **2. The Dataset**

The foundation of this predictive model is the **Medical Cost Prediction Dataset**, a comprehensive collection of patient records that link demographic profiles and clinical history to financial outcomes. 

* **Total Observations:** **$10,000$** unique patient records.
* **Target Variable:** `annual_medical_cost` — The total healthcare expenditure for the individual over a 12-month period.
* **Feature Categories:** The dataset contains 19 independent variables that capture a holistic view of the patient:
* **Demographics:** `age`, `gender`, and `city_type` (Urban, Semi-Urban, Rural).
* **Lifestyle & Activity:** `bmi`, `smoker` status, `physical_activity_level`, `daily_steps`, and `sleep_hours`.
* **Clinical History:** Presence of chronic conditions including `diabetes`, `hypertension`, `heart_disease`, and `asthma`.
* **Healthcare Utilization:** `doctor_visits_per_year`, `hospital_admissions`, `medication_count`, and `previous_year_cost`.
* **Financial Context:** `insurance_type` and `insurance_coverage_pct`.


#### **Key Empirical Findings from the Raw Data:**

* **Financial Range:** The annual medical costs in this dataset show significant variance, ranging from basic preventive care costs to high-expenditure cases exceeding **$\$30,000$**.
* **The "Legacy" Effect:** Initial inspection suggests a strong relationship between `previous_year_cost` and the target variable, indicating that past medical spending is often the best predictor of future trends.
* **Chronic Burden:** The dataset includes a high number of patients with overlapping conditions (comorbidities). For example, patients with both `diabetes` and `heart_disease` show a distinct cost distribution compared to those with no chronic conditions.

### **3. The Analytics Stack**

To build a high-performance financial forecasting tool, I used a modern Python-based data science stack. These tools were selected for their ability to handle large-scale data and implement complex ensemble learning algorithms.

* **Python:** The primary engine used for the entire analytical pipeline.
* **Pandas & NumPy:** These were my "workhorses" for data manipulation. They allowed me to clean $10,000$ records efficiently, handle missing values, and perform the vector calculations required for feature scaling.
* **Matplotlib & Seaborn:** I used these for **Visual Analytics**. By creating correlation heatmaps and distribution plots, I was able to see the relationship between factors like "Smoker Status" and "Annual Cost" before building the model.
* **Scikit-Learn (Sklearn):** The core machine learning library used for:
* **Data Pre-processing:** To convert categories like "Insurance Type" into numbers the model can understand.
* **Random Forest Regressor:** This was the primary algorithm. It was chosen because it creates Decision Trees to find the most accurate cost prediction.
* **Evaluation Metrics:** Used to calculate the **Mean Absolute Error (MAE)** and the **$R^{2}$ Score** to prove the model's reliability.


### **4. Data Wrangling & Feature Engineering**

Before moving to the predictive phase, the raw medical data underwent a rigorous transformation process. Financial forecasting requires extremely clean data because even a few missing values in categories like "Smoker" or "Insurance Type" can drastically skew the predicted annual costs.

#### **Data Cleaning & Imputation**

* **Handling Missing Values:** I identified that the `insurance_type` column had missing entries. Rather than deleting these records and losing valuable patient history, I imputed them with the value **'No Insurance'**, ensuring a complete clinical dataset.
* **Median Imputation for Vitals:** For any missing numerical data in vital signs, I used **Median Imputation**. This method is more robust against outliers (like extremely high medical bills) compared to using the average.

#### **Feature Encoding (Converting Text to Numbers)**

Since machine learning models only process numerical data, I transformed the categorical lifestyle and demographic variables:

* **Label Encoding:** Used for binary features like **Gender** and **Smoker Status**.
* **One-Hot Encoding:** Applied to complex categories like **Insurance Type** and **City Type**. This prevents the model from assuming a mathematical order between unrelated categories (e.g., it doesn't think "Urban" is "greater than" "Rural").

#### **Dimensionality & Scaling**

* **Comprehensive Feature Selection:** I included $19$ different features—ranging from `daily_steps` to `chronic_conditions`. This holistic approach allows the model to capture the "Total Health Picture" of the patient.
* **Target Stabilization:** During experimentation, I explored **Log Transformation** (using `np.log1p`) on the `annual_medical_cost` to normalize the target variable. This helps the model handle cases with extremely high costs (long-tail distributions) more effectively, though the final model was optimized to predict actual dollar amounts.

#### **Data Partitioning**

* **80/20 Train-Test Split:** I reserved $2,000$ patient records ($20\%$) as a "hold-out" set. The model never saw this data during training, providing a true test of how well it can predict the healthcare costs of a new patient.

By the end of this stage, the dataset was transformed from a raw spreadsheet into a high-performance feature matrix ready for the Random Forest algorithm.


```

```python
import pandas as pd

# Load the medical cost dataset
df = pd.read_csv('medical_cost_prediction_dataset.csv')

# 1. Basic Stats
total_avg_cost = df['annual_medical_cost'].mean()

# 2. Correlations with cost
# Encode smoker for correlation
df_corr = df.copy()
df_corr['smoker_num'] = df_corr['smoker'].map({'Yes': 1, 'No': 0})
# Focus on numerical and logical correlations
correlations = df_corr.select_dtypes(include=['number']).corr()['annual_medical_cost'].sort_values(ascending=False)

# 3. Impact of Smoker Status
smoker_impact = df.groupby('smoker')['annual_medical_cost'].mean()

# 4. Impact of Chronic Conditions (Average Cost)
chronic_cols = ['diabetes', 'hypertension', 'heart_disease', 'asthma']
chronic_impact = {}
for col in chronic_cols:
    impact = df.groupby(col)['annual_medical_cost'].mean()
    chronic_impact[col] = impact.to_dict()

# 5. Age and Cost Correlation
age_cost_corr = df['age'].corr(df['annual_medical_cost'])

print(f"Average Annual Cost: ${total_avg_cost:.2f}")
print("\nTop Correlations with Cost:\n", correlations.head(6))
print("\nSmoker vs Non-Smoker Average Cost:\n", smoker_impact)
print("\nChronic Condition Impacts (Means):")
for k, v in chronic_impact.items():
    print(f"{k}: {v}")



```

```text
Average Annual Cost: $8048.89

Top Correlations with Cost:
 annual_medical_cost    1.000000
hospital_admissions    0.355629
medication_count       0.133298
heart_disease          0.121740
previous_year_cost     0.089275
smoker_num             0.061673
Name: annual_medical_cost, dtype: float64

Smoker vs Non-Smoker Average Cost:
 smoker
No     7801.028240
Yes    8816.010442
Name: annual_medical_cost, dtype: float64

Chronic Condition Impacts (Means):
diabetes: {0: 7862.768114588592, 1: 8759.29402697495}
hypertension: {0: 7837.8739494382025, 1: 8570.557784722223}
heart_disease: {0: 7698.435367218465, 1: 10162.932742616033}
asthma: {0: 8028.406323594511, 1: 8240.860373443984}


```

### **5. Exploratory Data Analysis (EDA)**

In this phase, I conducted a deep dive into the $10,000$ patient records to uncover the primary drivers of healthcare spending. My goal was to move beyond the overall average of **$\$8,048.89$** to understand the specific triggers that cause costs to spike.

#### **A. The Primary Cost Drivers (Correlations)**

By calculating the statistical relationship between each patient feature and their annual bill, I identified the top "Red Flags" for high expenditure:

1. **Hospital Admissions:** **$0.36$ Correlation** — The single most significant driver; more frequent stays naturally lead to exponential cost increases.
2. **Medication Count:** **$0.13$ Correlation** — A high volume of recurring prescriptions indicates long-term management costs.
3. **Heart Disease:** **$0.12$ Correlation** — Among all clinical conditions, heart disease has the strongest direct link to financial burden.
4. **Previous Year Cost:** **$0.09$ Correlation** — Suggests that medical spending is often cyclical; high spenders one year are likely to remain high spenders the next.

#### **B. Lifestyle & Chronic Disease Impact**

I quantified how specific health factors move the needle on annual costs:

* **The "Smoker Premium":** * **Non-Smokers:** $\$7,801.03$ average annual cost.
* **Smokers:** $\$8,816.01$ average annual cost.
* *Insight:* Smoking adds an average of **$\$1,015$** in additional healthcare spending per year per patient.


* **The Chronic Disease Multiplier:**
* **Heart Disease:** Patients with heart disease pay an average of **$\$10,162$**, which is **$32\%$ higher** than those without the condition ($\$7,698$).
* **Diabetes:** Diagnosis increases the average bill by approximately **$\$900$** per year ($+11.4\%$).
* **Hypertension:** Adds roughly **$\$730$** to the annual baseline expenditure.


#### **C. Distribution & Outliers**

While the average sits at $\$8,048$, the data shows a "long-tail" distribution. This means while many patients stay within a predictable budget, a significant minority experience catastrophic costs exceeding **$\$25,000$**. Identifying the clinical profile of these "High-Cost" outliers is what makes the Random Forest model so valuable—it learns to spot the combination of factors (e.g., Age + Smoking + Diabetes) that leads to these financial peaks.

### **6. Machine Learning Strategy**

Moving from data exploration to predictive modeling, my strategy was to select an algorithm capable of capturing the "non-linear" nature of healthcare spending. Medical costs don't increase at a constant rate; instead, they often jump significantly when multiple risk factors (like smoking and high BMI) interact.

#### **A. Algorithm Selection: Random Forest Regressor**

I chose the **Random Forest Regressor** as the primary model for this project.

* **Why Random Forest?** A simple linear model assumes that each factor adds a fixed dollar amount to the bill. However, in medicine, being a "Smoker" might add $\$1,000$ to a healthy person's bill but $\$5,000$ to a patient with heart disease. Random Forest handles these complex "interactions" by building an ensemble of **$500$ different decision trees** and averaging their results to find the most stable prediction.

#### **B. Model Configuration & Hyperparameters**

To achieve the high $R^{2}$ score of $0.97$, I fine-tuned the model with the following technical specifications (as seen in Cell 26 of the notebook):

* **`n_estimators=500`**: The model combines the wisdom of 500 individual trees to reduce error.
* **`max_depth=20`**: Limits the complexity of each tree to prevent it from "memorizing" the training data (overfitting).
* **`min_samples_split=5` & `min_samples_leaf=2**`: These settings ensure the model only creates a new "branch" when there is enough data to support a statistically significant trend.

#### **C. Training & Validation Pipeline**

* **Hold-out Validation:** I used a standard **$80/20$ split**, training the model on $8,000$ records and validating it on $2,000$ "unseen" records. This confirms that the model can predict costs for a new patient it has never encountered before.
* **Error Measurement:** I used **Mean Absolute Error (MAE)** as the primary metric because it is easy for stakeholders to understand—it tells us exactly how many dollars off our prediction is on average.

By using this ensemble strategy, the model effectively balances high-level demographic trends with the specific clinical nuances of individual patient histories.

### **7. Key Outcomes & Model Performance**

The transition from data engineering to predictive modeling yielded exceptional results. By utilizing the **Random Forest Regressor**, the model successfully captured the intricate relationships between patient lifestyle, clinical history, and financial expenditure.

#### **Final Model Performance Metrics**

* **R² Score: $0.9789$** — This is a near-perfect score. It indicates that the model can explain approximately **$98\%$ of the variation** in annual medical costs. For an analyst, this proves that the selected features (like chronic conditions and previous costs) are highly representative of actual spending.
* **Mean Absolute Error (MAE): $\$633.12$** — On average, the model’s predictions are within roughly $\$633$ of the actual cost. Given that annual bills in the dataset can reach upwards of $\$30,000$, this margin of error is remarkably low, providing high confidence for financial forecasting.

#### **The "Engine" of the Prediction (Feature Importance)**

The Random Forest model allowed me to extract exactly which factors weigh the most when calculating a bill. This provides a clear hierarchy of what drives medical costs:

1. **Previous Year Cost:** The strongest indicator. It confirms that healthcare spending patterns are often persistent year-over-year.
2. **Hospital Admissions:** A direct physical driver; every admission acts as a major financial step-up in the annual total.
3. **Medication Count:** Reflects the complexity of the patient's health needs and recurring pharmacy expenses.
4. **Clinical Triggers (Heart Disease & Diabetes):** These chronic conditions act as significant multipliers, especially when combined with high BMI or smoking.

#### **Performance Stability**

By comparing the performance on the training data vs. the unseen test data, I confirmed that the model is not overfitting. The error rates remained consistent, proving that the model is **robust** and ready to be used on new, incoming patient data.

### **8. Challenges & Data Limitations**

 While the model achieved a high $R^{2}$ score of **$0.9789$**, there are specific real-world factors and technical hurdles that must be considered when interpreting these results.

* **The "Outlier" Impact:** Medical costs often follow a "long-tail" distribution. A single catastrophic health event (like a major surgery or long-term ICU stay) can result in costs that far exceed the predicted range. While the Random Forest model is robust, these rare "black swan" medical events remain difficult to forecast with $100\%$ precision.
* **Static vs. Dynamic Data:** The dataset provides a "snapshot" of a patient's health at a specific point in time. In reality, health is dynamic—a patient might quit smoking or develop a new chronic condition halfway through the year. The model currently cannot account for these **mid-year lifestyle changes**.
* **Regional Price Fluctuations:** Healthcare costs are heavily influenced by geography. A surgery in an urban private hospital may cost significantly more than the same procedure in a rural government facility. While we included `city_type`, the model is limited by the specific pricing structures present in this dataset and may need recalibration for different countries or healthcare systems.
* **Self-Reported Bias:** Features such as `physical_activity_level` and `sleep_hours` are often self-reported in medical surveys. This can introduce "social desirability bias," where patients overestimate healthy habits, potentially creating "noise" in the data that the model has to filter out.
* **Feature Completeness:** While we analyzed 19 features, certain high-impact variables were not available, such as **genetic predispositions**, **mental health history**, or **specific dietary habits**, all of which are known to influence long-term healthcare spending.


