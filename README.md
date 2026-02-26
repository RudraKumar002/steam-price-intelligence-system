# 🎮 Steam Price Intelligence System

An end-to-end machine learning system designed to analyze Steam game metadata and recommend optimal pricing strategies using only **pre-release information**.

---

## 📌 Project Overview

The **Steam Price Intelligence System** is built to simulate real-world deployment conditions where pricing decisions must be made **before a game is launched**.

Instead of relying on post-release engagement metrics, this system uses structured metadata and textual signals available at launch time to provide intelligent pricing recommendations.

---

## 🧠 Design Principles

- Only **pre-release metadata** is used.
- Post-release metrics (reviews, playtime, ownership, user counts, etc.) are removed to prevent data leakage.
- The system follows a **two-stage pricing decision pipeline**:
  1. Predict whether a game should be **Free or Paid**
  2. If Paid, recommend an appropriate **price tier**
- Structured and textual features are integrated for multi-modal learning.
- The workflow mirrors realistic ML deployment practices.

---

## 🎯 Objective

Develop a pricing intelligence system that:

1. Predicts whether a game should be **Free or Paid**
2. If Paid, recommends a suitable **price tier**

The goal is to move beyond simple price prediction and build a structured, data-driven pricing recommendation framework.

---

## 🚨 Problem Statement

Pricing is one of the most critical strategic decisions for indie game developers.

Games may be:

- Underpriced, leaving potential revenue unrealized  
- Overpriced, reducing sales volume  
- Priced without sufficient data-driven insight  

This project aims to:

- Analyze historical Steam game metadata
- Identify key pre-release features influencing pricing
- Engineer predictive structured features
- Integrate NLP-based textual signals
- Build classification models for intelligent pricing recommendations

---

## 📂 Dataset

- Size: ~400MB
- Not included in the repository due to GitHub size limitations.
- Available upon request.
- Place the dataset inside `data/raw/` before running notebooks.

---

## 🗂 Project Structure

steam-price-intelligence-system/
│   
├── notebooks/  
│ ├── 01_EDA.ipynb  
│ ├── 02_Feature_Engineering.ipynb  
│ └── 03_Modeling.ipynb 
│   
├── data/   
│ ├── raw/ (ignored in Git)     
│ └── processed/ (ignored in Git)    
│   
├── requirements.txt    
└── README.md   


---

## 📊 Phase 01 – Exploratory Data Analysis (Completed)

The exploratory analysis established a structured understanding of dataset composition, pricing patterns, and data quality.

### Key Observations

- The dataset contains ~95K records with no duplicate `appid` entries.
- Several columns contained high missingness and required removal.
- The price distribution is heavily right-skewed.
- Engagement and rating features exhibit long-tail behavior.
- The dataset contains both structured metadata and rich textual descriptions.

These insights informed the feature engineering strategy.

✅ Phase 01 Completed

---

## 📊 Phase 02 – Data Preprocessing & Feature Engineering (Completed)

This phase transformed the raw dataset into a modeling-ready format using only **pre-release predictive signals**.

### Key Steps

- Removed irrelevant and high-null columns.
- Dropped post-release engagement features to prevent leakage.
- Handled placeholder and invalid values (e.g., `-1`).
- Engineered structured features including:

  - `is_free`
  - `price_category`
  - Language counts
  - Developer & publisher counts
  - Developer tier classification
  - `release_year`
  - Package counts

- Performed multicollinearity analysis.
- Identified skewed distributions and applied `log1p` transformation.
- Finalized a clean, structured feature-engineered dataset.

The dataset is now aligned with real-world deployment constraints.

✅ Phase 02 Completed

---

## 🚀 Phase 03 – Model Training (In Progress)

Planned modeling stages:

- Baseline classification model
- XGBoost classifier
- Structured + TF-IDF feature integration
- Model comparison and evaluation
- Performance analysis and trade-offs

---

## 🛠 Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost(planned)
- Matplotlib / Seaborn
- TF-IDF (NLP)
- Sentence Transformers (planned)

---

## 🔮 Future Enhancements

- Transformer-based semantic embeddings
- Feature importance analysis
- Deployment-ready prediction pipeline

---

## 📎 Portfolio Note

This project is structured to reflect production-aware ML thinking, including:

- Data leakage prevention
- Multi-stage decision modeling
- Feature stability handling
- Repository discipline and reproducibility

---
