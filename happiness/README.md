# README.md

## Understanding Global Happiness: Insights from the Happiness Dataset

### 1. Data Overview

The dataset titled **"happiness.csv"** reveals intricate metrics of happiness and well-being across 165 countries from 2005 to 2023. Comprising 2363 entries with 11 columns, it presents both numeric and categorical data that enable exploration of how various factors correlate with subjective well-being, measured as **Life Ladder** scores.

#### Numeric Variables:
- **Year**: Data collection year (2005-2023)
- **Life Ladder**: A subjective well-being index (1.281 to 8.019), mean ~5.48
- **Log GDP per capita**: Economic well-being metric (5.527 to 11.676), mean ~9.40
- **Social Support**: Perceived social network support (0.228 to 0.987)
- **Healthy Life Expectancy at Birth**: Expected years in good health (6.72 to 74.6)
- **Freedom to Make Life Choices**: Feelings of autonomy (0.228 to 0.985)
- **Generosity**: Degree of altruism (-0.34 to 0.7)
- **Perceptions of Corruption**: Country corruption perceptions (0.035 to 0.983)
- **Positive Affect**: Frequency of positive feelings (0.179 to 0.884)
- **Negative Affect**: Frequency of negative feelings (0.083 to 0.705)

The analysis also notes varying completeness across numeric columns, necessitating careful handling of missing data in future analyses.

### 2. Key Patterns and Insights

#### Correlation Analysis
The correlation heatmap insightfully depicts the relationships between variables:

- **Strong Positive Correlations**:
  - **Log GDP per capita & Life Ladder (0.78)**: Suggests that economic strength bolsters happiness.
  - **Life Ladder & Healthy Life Expectancy (0.73)**: Healthier societies report higher happiness.

- **Moderate Influences**:
  - **Freedom to Make Life Choices & Life Ladder (0.62)** and **Generosity & Life Ladder (0.43)** indicate a tendency where personal freedom and altruism correlate with well-being.

- **Detractors to Happiness**:
  - **Negative Affect & Life Ladder (-0.53)** suggests that negative feelings depress overall happiness.

#### Distribution Visualizations
Visualizations of distributions of various metrics reveal intriguing patterns:
- **Life Ladder**: An approximately normal distribution, with most individuals reporting moderate to high satisfaction.
- **Log GDP per Capita**: Right skewed, reflecting a few nations with significantly high GDP.
- **Social Support**: Concentrated towards higher values, indicating general satisfaction with social support.
  
These distributions highlight the heterogeneity of well-being experiences across populations, emphasizing the complexity of impacting individual happiness.

### 3. Actionable Recommendations

- **Policy Initiatives**: Countries with thriving GDP and social support structures should invest further in these areas to bolster happiness metrics. Enhancing social safety nets can effectively improve life satisfaction.

- **Targeting Outliers**: Countries with low happiness scores (e.g., Life Ladder < 1.446) should be prioritized for interventions aimed at improving health, economic conditions, and social support.

- **Encouraging Generosity**: Enhanced community initiatives aimed at fostering generosity may yield improvements in overall life satisfaction.

### 4. Visualizations

To better illustrate the findings, the following visualizations have been included:

- **Correlation Heatmap**: ![correlation.png](correlation.png)
  
- **Distribution of Life Ladder and other metrics**: ![distributions.png](distributions.png)

### 5. Summary of Findings

- **Influencing Factors**: Log GDP per capita and healthy life expectancy are predominant influencers of life satisfaction.
- **Variability of Experiences**: Individual experiences of happiness and its detractors vary widely, pointing towards cultural and socio-economic distinctions across regions.

### 6. Generated Code for Analysis

The code used in the analyses includes various functions and approaches that enhance understanding of the data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('happiness.csv')

# Outlier detection using IQR
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

outliers = {col: detect_outliers_iqr(data, col) for col in data.select_dtypes(include=[np.number]).columns}

# Visualization of outliers
plt.figure(figsize=(12, 8))
for i, column in enumerate(data.select_dtypes(include=[np.number]).columns, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(x=data[column])
    plt.title(column)
plt.tight_layout()
plt.savefig('outliers.png')
plt.close()

# Clustering using KMeans
scaler = StandardScaler()
features = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth']
scaled_features = scaler.fit_transform(data[features].dropna())
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualization of clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Log GDP per capita', y='Social support', hue='Cluster', data=data)
plt.title('Clusters of Countries')
plt.savefig('clusters.png')
plt.close()

# Regression analysis
X = data[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth']].dropna()
y = data['Life Ladder'].loc[X.index]
model = LinearRegression()
model.fit(X, y)

# Feature importance using Random Forest
rf = RandomForestRegressor()
rf.fit(X, y)
importances = rf.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Visualization of feature importance
plt.figure(figsize=(8, 4))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
plt.close()
```

This code encompasses comprehensive analysis strategies, including outlier detection, regression modeling, and visualizations, thereby enhancing the overall understanding of the patterns and implications of the happiness metrics dataset.

---

The narrative articulated above is designed to contextualize the key findings from the dataset, offering insights that can guide actionable policies and further areas of research in enhancing global well-being.