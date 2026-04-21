---

## 🖥️ Streamlit Dashboard

The interactive dashboard allows you to:
- Select any customer by index
- View their **default risk probability**
- Inspect a **SHAP force plot** explaining the prediction**

```bash
streamlit run app.py
```

---

## 📈 Results Summary

| Metric | Value |
|---|---|
| ROC-AUC | 0.765 |
| Accuracy (thresh=0.2) | 89% |
| Recall (defaults) | 30% |
| Precision (defaults) | 29% |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)
![SHAP](https://img.shields.io/badge/SHAP-explainability-green)

- Python 3.12
- XGBoost
- scikit-learn
- imbalanced-learn (SMOTE)
- SHAP
- Streamlit
- SQLite (in-memory)
- Pandas

---

## 🔮 Future Work

- [ ] Hyperparameter tuning with Optuna
- [ ] Calibrate predicted probabilities (Platt scaling)
- [ ] Add LightGBM / CatBoost comparison
- [ ] Deploy dashboard to Streamlit Cloud
- [ ] Add customer segmentation clustering

---

## Acknowledgements

- [Home Credit Group](https://www.homecredit.net/) for the dataset
- [Kaggle Competition](https://www.kaggle.com/competitions/home-credit-default-risk)
- SHAP library by Scott Lundberg
