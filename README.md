##  House-Price-Prediction

**Predicting House Prices using a Stacking Ensemble of Regressors**
This project applies an ensemble learning technique to estimate house prices based on various housing features.

---

###  Key Highlights

* Achieved **R² Score: 0.998** on the test data
* Used **Stacking Regressor** combining multiple strong base models
* **Base Models:** Random Forest, Gradient Boosting, Extra Trees, Decision Tree, XGBoost
* **Meta Model:** Linear Regression
* Evaluation done using **cross-validation** and test set performance
* Final model saved using `joblib` for easy reuse

---

###  How to Predict on New Data

```python
import joblib
import pandas as pd

# Load new data
x_new = pd.read_csv('input.csv')

# Load trained model
model = joblib.load('final_stacking_model.joblib')

# Make predictions
predictions = model.predict(x_new)
```

---

### Download the Trained Model

Model file is large and hosted externally:
[Download stacking\_model.joblib]([https://drive.google.com/your-download-link](https://drive.google.com/drive/folders/17CwuvhEWMGfjBnpuBRzUpng36DxYl6s9?dmr=1&ec=wgc-drive-globalnav-goto))
