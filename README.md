
# 🧠 AI-Powered QSAR Modeling for Prostate Cancer Drug Discovery

This project applies **bioinformatics** and **machine learning** techniques to accelerate **drug discovery for prostate cancer**.
It leverages **ChEMBL bioactivity data**, **molecular descriptors**, and **AI models** to predict the potency of candidate molecules targeting the **Androgen Receptor (AR)** — a key protein in prostate cancer progression.


## 🚀 Project Overview

Prostate cancer remains one of the most prevalent cancers among men worldwide.
To reduce the time and cost of laboratory screening, this project builds a **QSAR (Quantitative Structure–Activity Relationship)** model that predicts **bioactivity (pIC50)** values directly from chemical structures (SMILES).

Using **Python** and **Machine Learning**, this pipeline provides a computational framework to identify promising drug candidates **in silico** before experimental testing.


## 🧩 Workflow Summary

1. **Data Collection:**

   * Extracted AR-targeted compounds and activity data from the **ChEMBL database**.
   * Filtered based on standard activity values (IC50).

2. **Data Preprocessing:**

   * Normalized IC50 values and converted to **pIC50**.
   * Removed missing or duplicate entries.

3. **Descriptor Calculation:**

   * Used **PaDEL-Descriptor** and **Lipinski descriptors** to generate physicochemical and structural features.

4. **Exploratory Data Analysis (EDA):**

   * Visualized feature distributions, correlations, and activity classes.

5. **Modeling:**

   * Trained and evaluated multiple regression models:

     * `AdaBoostRegressor`
     * `LGBMRegressor`
     * Other models benchmarked using **LazyRegressor**

6. **Evaluation Metrics:**

   * R² (Coefficient of Determination)
   * RMSE (Root Mean Square Error)
   * MAE (Mean Absolute Error)

7. **Deployment:**

   * Interactive web app (Flask) allowing users to input **SMILES** and receive **predicted pIC50**.


## 🧠 Example Use Case

1. User provides a compound’s **SMILES** string.
2. The model computes descriptors and predicts **bioactivity (pIC50)**.
3. Researchers can use these predictions to **prioritize compounds** for synthesis and in-vitro screening.

---

## 💻 Technologies Used

| Category              | Tools / Libraries       |
| --------------------- | ----------------------- |
| Programming           | Python 3.10+            |
| Data                  | ChEMBL Bioactivity      |
| Descriptor Generation | PaDEL-Descriptor, RDKit |
| Machine Learning      | scikit-learn, LightGBM  |
| Visualization         | matplotlib, seaborn     |
| Model Benchmarking    | LazyPredict             |
| Deployment            | Streamlit / Gradio      |

---

## ⚙️ Installation

Clone this repository:

```bash
git clone https://github.com/Hezekay/Postrate-cancer-drug-discovery-Bioinformatic-ML.git
cd prostate-cancer-drug-discovery
```

Create and activate a virtual environment:

```bash
python -m venv postrate_cancer_env
source postrate_cancer_env/bin/activate  # Linux/Mac
postrate_cancer_env\Scripts\activate     # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the App

To launch the Flask web app locally:

```bash
python app.py
```


Then open your browser at **[http://localhost:8501](http://localhost:8501)**


## 🧪 Sample Output

| SMILES                  | Predicted pIC50 |
| ----------------------- | --------------- |
| CCN(CC)CCOC(=O)c1ccccc1 | 6.78            |
| CCOC(=O)c1ccc(CN)cc1    | 5.91            |


## 📘 Disclaimer

⚠️ **This project is for research and educational purposes only.**
Predictions are **computational estimates** and **should not** be used for clinical or medical decisions without experimental validation.


## 🧑‍💻 Author

**Odetunde Hezkiah Oluwasegun**
Data Scientist & Machine Learning Engineer
📧 [odetundehezekiah@gmail.com](mailto:odetundehezekiah@gmail.com)
🔗Github:  https://github.com/Hezekay LinkedIn: www.linkedin.com/in/hezekiah-odetunde-65ab2b221 


## 🏷️ Citation

If you use this project in your research or publication, please cite it as:

> Odetunde H.O (2025). *Machine Learning–Based QSAR Modeling for Androgen Receptor Inhibitors in Prostate Cancer Drug Discovery.* GitHub Repository. https://github.com/Hezekay/Postrate-cancer-drug-discovery-Bioinformatic-ML.git


