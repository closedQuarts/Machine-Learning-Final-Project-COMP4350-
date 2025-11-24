Customer Lifetime Value (CLV) Prediction

This project builds a machine learning pipeline to predict Customer Lifetime Value (CLV) using real-world e-commerce transaction data. It includes data preprocessing, feature engineering, multiple model evaluations, hyperparameter optimization, visualization, and computational complexity analysis.

⸻

📁 Project Structure

├── data/                # Input dataset (CSV uploaded via Colab)
├── notebook/            # Colab notebook (.ipynb)
├── README.md            # Project documentation


⸻

📌 Objectives
	•	Predict Customer Lifetime Value (CLV)
	•	Engineer meaningful customer-level behavioral features
	•	Compare baseline and advanced ML models
	•	Optimize model performance using GridSearchCV
	•	Visualize predictions
	•	Analyze time and memory complexity

⸻

📥 1. Data Loading

The dataset is uploaded through Google Colab using:

from google.colab import files
uploaded = files.upload()
data = pd.read_csv(file_name, encoding='latin1')

Initial preprocessing includes:
	•	Converting InvoiceDate → datetime
	•	Creating TotalSpend = Quantity × UnitPrice
	•	Handling missing values

⸻

🛠️ 2. Feature Engineering

Customer-level aggregated features are generated:
	•	Recency – days since last purchase
	•	TransactionFrequency – number of unique invoices
	•	TotalSpend – total revenue
	•	UniqueProductsPurchased – product variety
	•	AvgSpendPerTransaction – average purchase value

These features form the core inputs for the ML models.

⸻

🤖 3. Modeling

Models trained:
	•	Random Forest Regressor
	•	Linear Regression (baseline)

Before training:

X_train, X_test, y_train, y_test = train_test_split(...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

Performance Metrics
	•	MAE (Mean Absolute Error)
	•	MSE (Mean Squared Error)
	•	R² Score

Random Forest outperformed Linear Regression, showing stronger predictive ability.

⸻

⚙️ 4. Hyperparameter Optimization

GridSearchCV tested multiple configurations of Random Forest:

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

Best Parameters:

{
  'n_estimators': 200,
  'max_depth': 10,
  'min_samples_split': 2,
  'min_samples_leaf': 4
}


⸻

📊 5. Visualization

A scatter plot compares actual vs predicted CLV:

plt.scatter(y_test, y_pred_best)
plt.plot(y_test, y_test, linestyle='--')

This visualization highlights prediction accuracy—points closer to the diagonal represent better performance.

⸻

⏱️ 6. Time & Space Complexity

Performance comparison:

Model	Training Time	Prediction Time	Memory Usage	Notes
Linear Regression	Very fast	Extremely fast	Low	Poor accuracy
Random Forest	Slower	Moderate	Higher	Best accuracy

Random Forest requires more computation but delivers substantially better results.

⸻

📌 Conclusion

This project demonstrates a complete Machine Learning workflow for CLV prediction.
Key outcomes:
	•	Feature engineering significantly boosts predictive power
	•	Random Forest is the most suitable model
	•	Hyperparameter tuning improves accuracy further
	•	Visual analysis supports model reliability

This pipeline can be adapted for real-world CLV prediction in e-commerce or CRM systems.

⸻

🚀 Future Improvements
	•	Add XGBoost & LightGBM comparisons
	•	Deploy the model via Flask/FastAPI
	•	Automate feature engineering
	•	Add cross-validation learning curves

⸻

📄 License

This project is free to use for educational and research purposes.

⸻
