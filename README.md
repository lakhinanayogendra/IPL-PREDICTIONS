# 🏏 IPL Match Winner Prediction using Machine Learning

This project predicts the **winner of an Indian Premier League (IPL) cricket match** using historical match data and a **Machine Learning model (Random Forest Classifier)**. The prediction is based on match features such as **batting team, bowling team, and venue**.

---

## 📌 Project Description

The Indian Premier League (IPL) is one of the most popular cricket leagues in the world. This project applies **Machine Learning techniques** to analyze historical IPL match data and predict the most probable winner of a match.

The system processes past IPL match data to calculate team-wise runs for each match and identifies winners based on total runs scored. Categorical features such as team names and venues are converted into numerical values using **Label Encoding** so they can be used by the machine learning model.

A **Random Forest Classifier** is trained on this processed dataset to learn patterns that influence match outcomes. Once trained, the model can predict the winning team for a given match scenario using inputs like batting team, bowling team, and venue.

---

## 🎯 Objectives

- Analyze historical IPL match data  
- Identify match winners based on runs scored  
- Train a machine learning model  
- Predict the winner of an IPL match  
- Measure model accuracy  

---

## 🧠 Methodology

- Data loading and preprocessing  
- Match-wise run aggregation  
- Winner identification logic  
- Feature encoding using Label Encoding  
- Train-test data splitting  
- Model training using Random Forest Classifier  
- Model evaluation using accuracy score  
