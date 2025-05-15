# Impact of AI on Digital Media (2020-2025)
```
Analyzing the global impact of AI-generated content on public perception, media.
https://www.kaggle.com/datasets/atharvasoundankar/impact-of-ai-on-digital-media-2020-2025
https://www.kaggle.com/code/iseedeep/mission-exploring-the-impact-of-ai
```

# Background
```
This dataset captures the impact of Artificial Intelligence (AI) adoption across various countries and industries between 2020 and 2025. It includes key indicators such as AI adoption rates, AI-generated content volumes, job loss percentages due to AI, revenue increases attributed to AI, human-AI collaboration rates, consumer trust in AI, and regulatory environments.

The purpose of collecting this data is to analyze how AI integration is transforming different sectors economically and socially. It helps researchers and policymakers understand trends in AI adoption, its benefits in terms of revenue growth, as well as potential risks like job displacement.

By studying these multi-dimensional factors, the dataset aims to provide a comprehensive overview of AI’s evolving role in the digital media landscape and broader industry contexts, enabling data-driven decisions for managing AI’s impact on society and the economy.
```

# Dataset is about
```
It covers critical metrics such as AI Adoption Rate, AI-Generated Content Volume, Job Loss Due to AI, Revenue Increase Due to AI, Human-AI Collaboration Rate, Top AI Tools Used, Regulation Status, Consumer Trust in AI, and Market Share of AI Companies. 

Key observations include:
1. Industry-Specific Trends: AI adoption varies significantly across sectors, with gaming, manufacturing, and automotive often showing higher adoption rates.

2. Geographic Differences: Countries like the USA, China, and the UK have diverse AI adoption rates, reflecting different levels of technological maturity and regulatory landscapes.

3. Impact on Jobs and Revenue: Job loss due to AI is generally higher in industries like legal and manufacturing, while sectors like marketing and retail benefit more in terms of revenue growth.

4. Top AI Tools: Tools like ChatGPT, Midjourney, Stable Diffusion, and Bard are prominently used, reflecting regional preferences.

5. Regulatory Influence: Stricter regulations often correlate with lower consumer trust, but not necessarily lower market share, indicating complex dynamics.
```

# EDA - What We Can Do with This Dataset

## 1. Industry-wise AI Adoption Rate
```
Objective:
To identify and compare the average adoption rate of AI technologies across different industries, highlighting which sectors lead in AI integration.

Approach:
I cleaned the dataset by removing missing values, then calculated the mean AI adoption rate for each industry. I used a bar plot to visualize the average adoption rates and a pie chart to present the proportional contribution of each industry to overall AI adoption.

Visualization:
I created a dual-plot figure with a bar chart on the left, showcasing the average AI adoption rate (%) for each industry, and a pie chart on the right, illustrating the proportion of AI adoption by industry for an intuitive sector-wise comparison.
```

## 2. Country-wise AI Impact on Job Loss and Job Gain
```
Objective:
To explore the impact of AI on job dynamics across different countries, focusing on both job loss due to AI automation and job gain through increased human-AI collaboration rates.

Approach:
I prepared the data by removing missing values from key columns, then calculated the average job loss and job gain (estimated using collaboration rates) for each country. I used heatmaps for a quick overview of the intensity of AI's impact and bar plots for a more detailed comparison of average rates across countries.

Visualization:
I created a 2x2 grid with heatmaps for job loss and job gain on the left, providing a clear intensity comparison, and bar plots on the right to highlight the average impact per country.
```

## 3. Correlation Between AI Adoption and Revenue Increase
```
Objective:
To investigate the relationship between AI adoption rates and the corresponding revenue increase, exploring the potential financial impact of AI investments across various industries.

Approach:
I cleaned the data by removing rows with missing values in the 'AI Adoption Rate (%)' and 'Revenue Increase Due to AI (%)' columns. I then calculated the correlation to quantify the strength of this relationship. For visualization, I used a scatter plot with a trend line to capture the overall correlation and a density-enhanced scatter plot to highlight the distribution of data points.

Visualization:
I created a 1x2 grid with a scatter plot featuring a trend line on the left, emphasizing the linear correlation, and a density-enhanced scatter plot on the right to reveal concentrated regions of data, providing a deeper understanding of the correlation pattern.
```

## 4. Consumer Trust in AI by Regulation Status
```
Objective:
To examine the impact of regulation on consumer trust in AI, highlighting the relationship between regulatory frameworks and public confidence in AI technologies.

Approach:
I filtered the data to exclude rows with missing 'Regulation Status' and 'Consumer Trust in AI (%)' values. I then grouped the data by regulation status to calculate the average trust levels. For visualization, I used a box plot to capture the distribution of trust scores and a bar plot to showcase the average trust for each regulation category. Outliers were also identified and annotated for further insights.

Visualization:
I created a 1x2 grid: the left panel features a box plot for distribution analysis, including median lines and outlier annotations, while the right panel presents a bar plot with average trust levels, enhanced with direct value annotations for clarity.
```

## 5. Top AI Tools Analysis
```
Objective:
To explore the most commonly used AI tools in the dataset, identifying their frequency of usage and overall popularity to gain insights into the most preferred tools in the industry.

Approach:
I preprocessed the data to remove rows with missing values in the 'Top AI Tools Used' column. The tools were then cleaned, tokenized, and counted to generate a frequency distribution. For visualization, I used a combination of bar plots, word clouds, pie charts, and treemaps to effectively capture the most popular tools.

Visualization:
The analysis was split into two groups for better coverage:

Group 1: Bar plot (top 10 most frequent tools) and word cloud for a broader visual representation.

Group 2: Pie chart and treemap to highlight the relative proportions and hierarchical structure of the top 10 tools.
```

## 6. Visualizing AI Content by Country (TBs per year)
```
Objective:
To visualize the global distribution of AI-generated content volume across different countries to identify the leading contributors and their relative contributions to the total volume.

Approach:
I first grouped the data by country, summing the AI-generated content volume for each country to capture the overall impact. I then used a choropleth map with the 'country names' location mode to accurately plot the distribution on a world map, applying the 'Viridis' color scale for enhanced visual contrast.

Visualization:
I used Plotly Express to create a choropleth map, providing an interactive global view of AI-generated content volumes by country, helping to identify the major players in AI content production.
```

## 7. Correlation Heatmap
```
Objective:
To examine the relationships between key numeric metrics in the digital media industry, such as AI adoption rates, job loss percentages, and revenue increases, to uncover potential patterns and insights.

Approach:
I selected only the relevant numeric columns from my dataset and calculated their pairwise correlation matrix. I then visualized this matrix as a heatmap using the 'coolwarm' color palette to highlight both positive and negative correlations, ensuring that each relationship is clearly represented.

Visualization:
I used a Seaborn heatmap with annotations for precise correlation values, providing a quick, intuitive view of the interconnectedness between key AI impact metrics.
```


# Algorithms, Pre-Processing, Evaluation Metrics

## 1. Industry-wise AI Adoption Rate 
```
I am doing this to predict the AI Adoption Rate (%) based on factors like AI-generated content volume, job loss due to AI, revenue increase, human-AI collaboration, and consumer trust. Understanding these relationships helps reveal what drives AI adoption across industries and countries.

By training and comparing multiple regression models, I aim to find the most accurate model for predicting AI adoption. This insight can guide businesses and policymakers in strategizing AI integration and anticipating its impacts more effectively.

1. Linear Regression
Why: It’s a simple and interpretable model for predicting continuous outcomes like AI adoption rate.
Strength: Helps understand linear relationships between features and the target variable.
Use case: Good baseline to check how well a linear model fits the data.

2. Random Forest Regressor
Why: It captures complex, nonlinear relationships and interactions between features.
Strength: Robust to outliers and overfitting; often improves prediction accuracy on diverse datasets.
Use case: Useful when data patterns are complex and can’t be captured by linear models.

3. Lasso Regression
Why: It performs feature selection by shrinking less important feature coefficients to zero through regularization.
Strength: Helps reduce model complexity and improve generalization, especially with correlated features.
Use case: Ideal when you want a simpler, more interpretable model that focuses on key predictors.

Here’s the pre-processing being used and why:
1. Handling Missing Values: Rows with missing critical values in 'Industry' and 'AI Adoption Rate (%)' are dropped to ensure data quality and avoid errors during modeling.

2. Cleaning Categorical Data: The 'Industry' column values are stripped of extra spaces and standardized to title case to maintain consistency and reduce redundant categories.

3. Outlier Removal: Z-score method is used to remove extreme outliers in 'AI Adoption Rate (%)', improving model accuracy by preventing skew from anomalous data points.

4. Feature Scaling: StandardScaler is applied to numerical features to standardize their range, which helps models converge faster and perform better, especially those sensitive to feature scales like linear models and regularized regressors.

Overall purpose:
The preprocessing steps standardize the data and convert categorical data into a numeric form, making it suitable for machine learning algorithms and improving model performance and reliability.

Evaluation Metrics and Why We Use Them
1. Mean Absolute Error (MAE)
Why: MAE measures the average absolute difference between actual and predicted values. It gives a straightforward interpretation of average prediction error in the same units as the target variable.
Use: Useful to understand the typical size of errors without exaggerating large errors.

2. Mean Squared Error (MSE)
Why: MSE calculates the average of squared differences between actual and predicted values, penalizing larger errors more heavily than MAE.
Use: Good for highlighting models with large prediction errors, useful when large errors are particularly undesirable.

3. Root Mean Squared Error (RMSE)
Why: RMSE is the square root of MSE, bringing the error metric back to the original units of the target variable, making it easier to interpret than MSE.
Use: Commonly used to compare prediction errors in a meaningful scale, emphasizing larger errors.

4. R-squared (R²)
Why: R² indicates the proportion of variance in the dependent variable that is predictable from the independent variables. Values closer to 1 mean better explanatory power.
Use: Standard metric for assessing how well the model fits the data.

5. Adjusted R-squared
Why: Adjusted R² adjusts the R² value based on the number of predictors and sample size, penalizing unnecessary complexity.
Use: More reliable when comparing models with different numbers of features, avoiding overfitting.

Summary:
These metrics together provide a comprehensive picture of model performance — how accurate the predictions are (MAE, MSE, RMSE) and how well the model explains the variation in the data (R², Adjusted R²). Using multiple metrics ensures robust evaluation and helps choose the best predictive model.
```

## 2. Country-wise AI Impact on Job Loss
```
I am doing this to predict whether AI will cause significant job loss (over 20%) in different countries and industries based on various factors like AI adoption, collaboration rates, revenue impact, and regulations. This helps understand the risk of job displacement due to AI and identify patterns or conditions linked to high job loss.

By training and comparing different machine learning models, we aim to find the best predictive model that can reliably classify situations with high job loss risk. This can inform policymakers, businesses, and stakeholders to make better decisions about AI adoption, workforce planning, and regulation.

1. Logistic Regression
Why: It’s a simple, interpretable model for binary classification tasks like this one (predicting job loss: yes/no).
Strength: It provides probabilities and clear insights into how each feature influences the outcome.
Use case: Good baseline model to see if linear relationships exist between features and job loss.

2. Decision Tree Classifier
Why: It captures non-linear relationships and interactions between features without requiring feature scaling.
Strength: Easy to visualize and interpret, showing decision rules clearly.
Use case: Helps understand complex decision boundaries and feature importance, plus good for explaining model behavior.

3. K-Nearest Neighbors (KNN)
Why: It’s a simple, instance-based learner that makes predictions based on similarity to training examples.
Strength: No explicit training phase; adapts well to local data structure.
Use case: Good to check if proximity in feature space correlates with job loss risk, offering a different perspective from parametric models.

In summary: These models are chosen because they represent diverse approaches — linear (Logistic Regression), rule-based (Decision Tree), and instance-based (KNN) — allowing a robust comparison to find the best fit for this dataset.

Here’s the pre-processing being used and why:

1. StandardScaler for Numerical Features
What: It scales numerical features to have zero mean and unit variance.
Why: Many machine learning models (like Logistic Regression and KNN) perform better and converge faster when features are on a similar scale. It prevents features with larger numeric ranges from dominating the model.

2. OneHotEncoder for Categorical Features
What: It converts categorical variables (like 'Country', 'Industry', 'Regulation Status') into binary columns for each category.
Why: Most ML algorithms can’t handle non-numeric data directly. One-hot encoding allows the model to treat each category as a separate feature without implying any ordinal relationship.

3. ColumnTransformer
What: It applies the appropriate transformations to the corresponding feature sets (numerical or categorical) in a single step.
Why: It keeps the pipeline clean and efficient by applying transformations only where needed, ensuring the data is ready for the model in one go.

Overall purpose:
The preprocessing steps standardize the data and convert categorical data into a numeric form, making it suitable for machine learning algorithms and improving model performance and reliability.

Here are the evaluation metrics used and why each is important:

1. Accuracy
What: The proportion of total correct predictions (both true positives and true negatives) out of all predictions.
Why: It gives an overall measure of how often the model is correct. Useful when classes are balanced.

2. Precision
What: The proportion of true positive predictions out of all positive predictions made by the model.
Why: Important when the cost of false positives is high — it tells us how reliable positive predictions are.

3. Recall (Sensitivity)
What: The proportion of actual positives correctly identified by the model.
Why: Crucial when missing a positive case is costly (false negatives). Measures the model’s ability to detect all positives.

4. F1 Score
What: The harmonic mean of precision and recall.
Why: Balances precision and recall into a single metric, useful when you want to find a good trade-off between false positives and false negatives.

5. Confusion Matrix
What: A table showing counts of true positives, true negatives, false positives, and false negatives.
Why: Helps visualize how the model is performing across classes and understand the types of errors it makes.

Summary:
These metrics collectively give a detailed picture of the model’s performance, especially in a classification problem where both false positives and false negatives matter. The F1 score is particularly valuable if the classes are imbalanced or if you need a balance between precision and recall.
```