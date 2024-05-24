# Customer_Churn_Telco_Analysis

Project Overview

Objective: Developed a predictive model to identify customers likely to churn in a telecom company.

Data: Utilized customer data, including demographics, usage patterns, billing information, and customer service interactions.

Features: Engineered key features such as tenure, monthly charges, contract type, payment method, and number of customer service calls.

Techniques and Models Used
Logistic Regression:

Description: Implemented logistic regression as a baseline model for churn prediction.

Process: Conducted data preprocessing (handling missing values, scaling features), and feature selection.

Evaluation Metrics: Analyzed performance using accuracy, precision, recall, F1-score, and ROC-AUC.

Advantages: Simple, interpretable, and quick to train.

Disadvantages: Limited in capturing complex relationships and interactions between features.

Artificial Neural Networks (ANN):

Description: Built and trained an ANN model for churn prediction.

Architecture: Designed a multi-layer perceptron (MLP) with input, hidden, and output layers.

Process: Performed hyperparameter tuning (number of layers, number of neurons per layer, activation functions, learning rate).

Evaluation Metrics: Assessed performance using the same metrics as logistic regression (accuracy, precision, recall, F1-score, ROC-AUC).

Advantages: Capable of capturing complex nonlinear relationships and interactions between features.

Disadvantages: Requires more computational resources, longer training times, and is less interpretable compared to logistic regression.
Power BI Dashboard

Objective: Created an interactive Power BI dashboard to visualize and monitor customer churn metrics.
Components:

Overview Page: Summary statistics of the dataset, including total customers, churn rate, and key metrics.

Churn Prediction Insights: Visualizations of predicted churn probabilities, segmented by key features such as contract type, payment method, and tenure.

Customer Segmentation: Clustered visualizations displaying high-risk customer segments, enabling targeted retention strategies.

Results and Insights

Performance Comparison:

Logistic Regression:
Achieved an accuracy of 78%,  ROC-AUC score of 83%.
Provided clear insights into feature importance and customer behavior patterns.

ANN:
Achieved an accuracy of 78%, precision of 83%, recall of 86% .
Demonstrated superior performance in terms of capturing complex patterns, resulting in higher predictive accuracy and recall.
Key Insights:
Identified high-risk customer segments based on key features such as long customer service call durations and high monthly charges.
Recommended targeted retention strategies based on model predictions, potentially reducing churn rates by E%.
Dashboard Insights:
Enabled real-time monitoring of churn metrics and model performance.
Provided actionable insights through interactive visualizations, aiding decision-making for marketing and customer service teams.
Technical Skills Demonstrated
Data Preprocessing: Data cleaning, feature engineering, and normalization.
Modeling: Logistic regression, ANN, hyperparameter tuning, model evaluation.
Visualization: Interactive dashboards, data visualization, storytelling with data.
Tools: Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, Power BI.
Soft Skills Demonstrated
Problem-Solving: Identified and addressed the challenge of predicting customer churn.
Analytical Thinking: Compared and evaluated different modeling approaches to find the most effective solution.
