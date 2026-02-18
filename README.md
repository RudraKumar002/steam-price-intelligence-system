# 🎮 Steam Price Intelligence System

## 📌 Project Overview

The **Steam Price Intelligence System** is an end-to-end machine learning project designed to analyze Steam game metadata and recommend optimal pricing strategies for indie game developers.

This system combines structured features (genre, reviews, release data, playtime, etc.) with natural language processing on game descriptions to understand how different factors influence game pricing in the Steam marketplace.

The goal is to move beyond simple price prediction and develop a data-driven pricing recommendation framework.

---

## 🎯 Problem Statement

Pricing is one of the most critical decisions for indie game developers.

Many games may be:

* Underpriced, leaving potential revenue unrealized
* Overpriced, reducing sales volume
* Priced without sufficient data-driven insight

The objective of this project is to:

* Analyze historical Steam game data
* Identify key features that influence pricing
* Build regression and classification models
* Provide an intelligent price recommendation range based on game attributes

This system aims to help developers make strategic pricing decisions backed by machine learning insights.

---

## 📊 Phase 01 – Exploratory Data Analysis (Completed)

The exploratory analysis provided a structured understanding of the Steam dataset’s composition, pricing patterns, and data quality.

### Key Observations

* The dataset contains **94,948 records with no duplicate `appid` entries**, ensuring entity-level integrity.
* Several columns contain **substantial missing values**, with some features exceeding 50% nulls and requiring removal or careful preprocessing.
* The **price distribution is highly right-skewed**, dominated by low-cost and free games, with a small number of extreme outliers.
* Engagement and rating features exhibit **long-tail distributions and placeholder values**, indicating the need for data cleaning and transformation.
* The dataset combines **structured numerical features and rich textual metadata**, supporting both regression and transformer-based NLP modeling.

These insights establish a strong foundation for the next stage of development.

### ✅ Phase 01 Completed

➡ Moving to **Phase 02 – Data Preprocessing & Feature Engineering**

---