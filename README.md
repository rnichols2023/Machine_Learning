# Machine_Learning

## 🚲 Bike Sharing Demand — Machine Learning Project
This project analyzes the Bike Sharing Dataset (daily data) to understand the factors that influence bike rental demand and to build a predictive machine learning model. The workflow includes data cleaning, preprocessing, exploratory data analysis (EDA), feature engineering, and a multiple linear regression model.

## 📁 Dataset Overview
The dataset contains 731 daily observations from 2011–2012, with weather, calendar, and usage information.

instant – Unique record index  
dteday – Date of the observation  
season – Season of the year (1–4)  
yr – Year (0 = 2011, 1 = 2012)  
mnth – Month (1–12)  
holiday – Whether the day is a holiday  
weekday – Day of the week (0–6)  
workingday – Whether the day is a working day  
weathersit – Weather situation category  
temp – Normalized temperature  
atemp – Normalized “feels like” temperature  
hum – Normalized humidity  
windspeed – Normalized wind speed  
casual – Count of casual (non‑registered) users  
registered – Count of registered users  
cnt – Total bike rentals (target variable)  

## 📊 Exploratory Data Analysis (EDA)
Several visualizations were created to understand relationships between features and bike demand.

## 📊 Key Visualizations & Insights

### **Boxplots for Categorical Variables**
- Showed how demand varies by season, year, month, weekday, holiday, and weather.  
- Higher demand observed in warmer seasons and non‑holiday working days.  

### **Regression Plots for Continuous Variables**
- `temp` and `atemp` showed a strong positive relationship with demand.  
- `hum` and `windspeed` showed negative relationships with demand.  
- These trends align with real‑world expectations: people ride more when it’s warm and less when it’s humid or windy.  

### **Distribution Patterns**
- Demand (`cnt`) showed clear seasonal patterns and year‑over‑year growth.  
- Visualizations helped identify which features were most influential before modeling.  

## 🤖 Machine Learning Model

A **Multiple Linear Regression** model was built using the following predictors:

- temp  
- hum  
- windspeed  
- workingday  

**Target Variable:**  
- cnt (total bike rentals)

### **Modeling Steps**
- Split data into training and validation sets (60/40).  
- Fit a linear regression model using scikit‑learn.  
- Extracted coefficients to understand feature influence.  
- Evaluated performance using:  
  - RMSE  
  - MAE  
  - MAPE  
  - Residual analysis  

### **Model Findings**
- Temperature had the strongest positive effect on bike demand.  
- Humidity and windspeed negatively impacted demand.  
- Working days slightly increased demand.  
- Results matched intuitive expectations about bike usage patterns.

## 📈 Model Performance

The regression model produced reasonable error metrics given the simplicity of the approach.

This project demonstrates:
- Feature engineering  
- Scaling  
- Dummy encoding  
- Regression modeling  
- Model evaluation  
- Interpretation of coefficients  

## 🧠 Key Takeaways

- Weather strongly influences bike rental demand.  
- Warmer temperatures increase usage, while humidity and wind reduce it.  
- Demand patterns differ across seasons, months, and weekdays.  
- A simple linear regression model can capture meaningful relationships in the data.  

## 📦 Repository Contents

- `Nichols_Rachel_FinalProject.ipynb` — Full analysis and model  
- `day.csv` — Dataset  
- Regression plots (`regression_temp.png`, etc.)  
- This README  

## 🙌 Final Notes

This project showcases practical machine learning skills, including data preprocessing, visualization, and regression modeling. It is suitable for academic submission, portfolio use, and GitHub demonstration.  

