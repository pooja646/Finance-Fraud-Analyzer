# Finance-Fraud-Analyzer
A PySpark based modular pipeline designed for end-to-end financial fraud detection

Setup Instructions
Follow the steps below to set up and run the PySpark-Based Financial Fraud Detection Pipeline from the GitHub repository.

1. Clone the Repository
```bash
  git clone https://github.com/your-username/financial-fraud-detection.git
  cd financial-fraud-detection
```

2. Set Up Data Directory
Create a data folder in the project root to store uploaded or sample CSV files:

```bash
mkdir data
```
```bash
curl -L -o data/synthetic_fraud_transactions_dataset.csv https://path-to-your-dataset.csv
```
3. Run the Application
Start the Streamlit app:

````bash
streamlit run streamlit_app.py
````
By default, the app will open in your browser at:

```
arduino
http://localhost:8501
```
