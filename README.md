# Impact of AI on Digital Media (2020-2025)
```
Analyzing the global impact of AI-generated content on public perception, media.
https://www.kaggle.com/datasets/atharvasoundankar/impact-of-ai-on-digital-media-2020-2025
```

## About Dataset
### About the Dataset
```
This dataset explores the influence of AI-generated content across various industries, including journalism, social media, entertainment, and marketing. It provides insights into public sentiment, engagement trends, economic impact, and regulatory responses over time.

With AI-generated content becoming increasingly prevalent, this dataset serves as a valuable resource for data analysts, business strategists, and machine learning researchers to study trends, detect biases, and predict future AI adoption patterns.
```

```
1. What we are going to do/Background
2. What is dataset is about
3. Steps to do
[Data Collection (Done)] → [Preprocessing (Cleaning)] → [Model Training] → [Prediction] → [Evaluation] 
``` 


# Understanding the Dataset: "Impact of AI on Digital Media (2020-2025)"
## 1. Topic Overview
```
This dataset explores how Artificial Intelligence (AI) is transforming the digital media industry between 2020 and 2025. It likely covers:
1. AI adoption trends in different media sectors (social media, streaming, journalism, advertising, etc.)
2. Economic and operational impacts (revenue changes, job displacement, efficiency improvements)
3. Key AI technologies used (machine learning, NLP, computer vision, generative AI)
4. Case studies of companies leveraging AI in media
```

## 2. What the Dataset Likely Contains
```
Based on the description, the dataset may include:
1. Time-series data (2020–2025) tracking AI adoption
2. Sector-wise breakdown (e.g., social media vs. news vs. entertainment)
3. Performance metrics (engagement, revenue, cost savings)
4. AI technology usage (e.g., recommendation systems, automated content generation)
5. Forecasts & predictions for 2025
```

## 3. What We Can Do with This Dataset
### a. Exploratory Data Analysis (EDA)
```
1. Trend Analysis: How has AI adoption grown over time?
2. Sector Comparison: Which media sectors use AI the most?
3. Impact Assessment: Does AI increase revenue or reduce jobs?
4. Correlation Studies: Is AI adoption linked to higher user engagement?
```

### b. Visualization
```
1. Line charts: AI adoption growth (2020–2025)
2. Bar plots: AI impact across different sectors
3. Heatmaps: Correlation between AI usage and business metrics
4. Pie charts: Breakdown of AI technologies used
```

### c.Predictive Analysis (If Data Allows)
```
1. Forecast AI adoption in 2025
2. Predict which sectors will be most disrupted
3. Machine learning models to estimate future impacts
```

### d. Insights & Reporting
```
1. Key takeaways: How AI is reshaping digital media
2. Business implications: Should companies invest more in AI?
3. Ethical concerns: Job losses, misinformation risks
```

# Exploratory Data Analysis (EDA) for "Impact of AI on Digital Media 2020-2025" Dataset
```
Based on the dataset from Kaggle about AI's impact on digital media, here are the key EDA approaches we could take:

1. Basic Data Inspection
- Check dataset structure (rows, columns)
- Examine data types of each column
- Identify missing values
- Look for duplicate entries
- Summary statistics for numerical columns

2. Temporal Analysis
- Analyze trends over time (2020-2025)
- Compare year-over-year changes in key metrics
- Identify seasonal patterns if quarterly/monthly data exists

3. Industry Sector Analysis
- Breakdown by digital media sectors (social media, streaming, news, etc.)
- Compare AI adoption rates across different sectors
- Identify which sectors are most/least impacted by AI

4. AI Technology Analysis
- Examine different AI technologies used (ML, NLP, computer vision, etc.)
- Adoption rates of specific AI technologies
- Effectiveness of different AI approaches

5. Impact Metrics Analysis
- Changes in productivity metrics
- Revenue impact analysis
- User engagement changes
- Content production metrics

6. Geographic Analysis (if location data exists)
- Regional differences in AI adoption
- Compare impacts across different markets

7. Correlation Analysis
- Relationships between AI adoption and business metrics
- Identify which AI applications correlate most with success

8. Visualization Opportunities
- Time series charts for trends over years
- Bar charts comparing sectors/technologies
- Heatmaps for correlation analysis
- Box plots for distribution of impact metrics

9. Outlier Detection
- Identify unusual cases (extremely high/low impact)
- Examine potential data errors

10. Predictive Potential
- Assess whether the data could support predictive modeling
- Identify gaps that would prevent forecasting
```

# AI in Digital Media: Algorithms, Preprocessing & Evaluation Metrics
## 1. Algorithms for Analysis
```
Depending on the dataset structure, different machine learning algorithms can be applied:
A. Predictive Modeling (Regression/Time Series)
i. Linear Regression
- Predicts continuous values (e.g., revenue growth, AI adoption rate).
- Justification: Simple, interpretable, works well for linear trends.

ii. Random Forest / XGBoost Regression
- Handles non-linear relationships (e.g., AI impact on engagement).
- Justification: Robust to outliers, captures feature importance.

iii. ARIMA / Prophet (Time Series Forecasting)
- Forecasts AI adoption trends (2020–2025).
- Justification: Best for temporal data with seasonality.

B. Classification (If Categories Exist)
i. Logistic Regression / Decision Trees
- Classifies media sectors by AI impact (High/Medium/Low).
- Justification: Good for binary/multiclass problems.

ii. Clustering (K-Means, DBSCAN)
- Groups companies/sectors by AI adoption patterns.
- Justification: Finds hidden segments in data.
```

## 2. Data Preprocessing Steps
```
Before applying algorithms, the data must be cleaned and structured:

A. Handling Missing Data
- Drop or Impute (mean/median for numerical, mode for categorical).
- Justification: Ensures no bias in model training.

B. Feature Engineering
i. Normalization (Min-Max / Z-Score)
- Needed for distance-based algorithms (K-Means, SVM).

ii. One-Hot Encoding
- Converts categorical variables (e.g., media sector) into numerical.

C. Time-Series Processing
i. Resampling (Yearly → Quarterly if needed)
- Lag Features (for forecasting models like ARIMA).

D. Outlier Removal
i. IQR / Z-Score methods
- Removes extreme values distorting trends.
```

## 3. Evaluation Metrics
```
The right metrics depend on the problem type:

A. Regression (Predicting Continuous Values)
i. Mean Absolute Error (MAE)
- Measures average prediction error (easy to interpret).

ii. R² Score
- Explains variance captured by the model (0–1 scale).

B. Classification (Categorical Outcomes)
i. Accuracy / F1-Score
- Evaluates model performance in class prediction.

ii. Precision-Recall Curve
- Useful if classes are imbalanced.

C. Clustering (Unsupervised Learning)
i. Silhouette Score
- Measures cluster separation quality.

ii. Elbow Method (WCSS)
- Helps choose optimal K in K-Means.

D. Time-Series Forecasting
i. Mean Squared Error (MSE)
- Penalizes large errors more.

ii. MAPE (Mean Absolute Percentage Error)
- Easy to interpret in business terms.
```

## 4. Justification Summary
```
     Task	                Best Algorithm	                Preprocessing Needed	             Evaluation Metric
Trend Prediction	       ARIMA / Prophet	               Lag features, resampling	                MSE, MAPE
Sector Comparison	   Random Forest / XGBoost	           One-hot encoding, scaling	       R², Feature Importance
Adoption Forecasting	  Linear Regression	             Outlier removal, normalization	              MAE, R²
Sentiment Analysis	        BERT / TF-IDF	             Text cleaning, tokenization	           F1-Score, Accuracy
```