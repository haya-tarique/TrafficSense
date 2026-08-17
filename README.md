\# 🚦 TrafficSense



\### AI-Powered Traffic Intelligence \& Forecasting Dashboard



TrafficSense is a machine-learning-based traffic forecasting system that predicts traffic intensity and classifies congestion levels using historical traffic patterns.



The project combines \*\*XGBoost regression, feature engineering, SHAP explainability, and an interactive Streamlit dashboard\*\* to provide understandable traffic insights.



\---



\## ✨ Features



\- 📊 Interactive traffic dashboard

\- 🤖 XGBoost traffic prediction

\- 🚦 LOW / MEDIUM / HIGH congestion classification

\- 📈 Traffic trend visualization

\- 🔍 Actual vs Predicted analysis

\- 🧠 SHAP-based prediction explainability

\- 📌 Feature importance analysis

\- 📊 Model performance comparison

\- 🎛️ Interactive observation selection

\- 🌙 Professional dark dashboard UI

\- 📱 Responsive Streamlit layout



\---



\## 🧠 Machine Learning



TrafficSense uses engineered historical traffic features:



\- Last Traffic Value

\- Average Traffic

\- Traffic Variation

\- Minimum Traffic

\- Maximum Traffic



The primary prediction model is:



\*\*XGBoost Regression\*\*



The model predicts a continuous traffic intensity value.



The predicted value is then converted into a congestion category.



\### Congestion Classification



| Traffic Intensity | Classification |

|---|---|

| `< 0.1971` | 🟢 LOW |

| `0.1971 – 0.4040` | 🟡 MEDIUM |

| `> 0.4040` | 🔴 HIGH |



\---



\## 📊 Model Performance



The evaluated models include:



| Model | R² | RMSE | MAE |

|---|---:|---:|---:|

| Linear Regression | 0.7328 | 0.1057 | 0.0766 |

| Random Forest | 0.8328 | 0.0836 | 0.0602 |

| XGBoost | \*\*0.8437\*\* | \*\*0.0809\*\* | \*\*0.0618\*\* |



XGBoost achieved the strongest overall R² and RMSE among the evaluated models.



\---



\## 🔍 Explainable AI



TrafficSense uses \*\*SHAP (SHapley Additive exPlanations)\*\* to explain individual model predictions.



The SHAP section shows:



\- Feature contribution

\- Feature value

\- Positive or negative impact

\- Strongest contributing feature

\- Prediction-specific explanation



This makes the machine-learning predictions easier to interpret.



\---



\## 🖥️ Dashboard Sections



\### Overview



Provides a high-level view of:



\- Model performance

\- Current traffic prediction

\- Congestion status

\- Traffic signals

\- Traffic activity



\### Traffic Forecast



Allows users to select an individual observation and view:



\- Predicted traffic intensity

\- Actual traffic value

\- Prediction error

\- Congestion level

\- Model input features

\- AI interpretation



\### Analytics



Provides:



\- Congestion distribution

\- Traffic statistics

\- Traffic trend

\- Actual vs predicted comparison

\- Dataset-level insights



\### Model Insights



Provides:



\- XGBoost performance

\- Model comparison

\- Feature importance

\- SHAP explainability

\- Prediction breakdown

\- Feature descriptions

\- Model decision explanation



\### About



Provides information about the project, technology stack and prediction pipeline.



\---



\## 🛠️ Technology Stack



\### Programming



\- Python



\### Machine Learning



\- XGBoost

\- Scikit-learn

\- SHAP



\### Data Processing



\- NumPy

\- Pandas



\### Visualization



\- Altair



\### Dashboard



\- Streamlit



\### Model Serialization



\- Joblib



\---



\## 📁 Project Structure



```text

TrafficSense/

│

├── app.py

├── requirements.txt

├── README.md

│

├── data/

│   └── processed\_features.npz

│

├── models/

│   └── xgboost\_traffic\_model.pkl

│

└── venv/

