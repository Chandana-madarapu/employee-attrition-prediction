import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(page_title="Attrition Predictor", layout="wide")

st.title("💼 Employee Attrition Prediction System")

# Load data
df = pd.read_csv("HR-Employee-Attrition.csv")

# Encode target
df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})

# Encode categorical
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split
X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ======================
# SIDEBAR INPUT
# ======================
st.sidebar.header("Enter Employee Details")

age = st.sidebar.slider("Age", 18, 60, 30)
income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
years = st.sidebar.slider("Years at Company", 0, 40, 5)
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])

# Convert input
overtime = 1 if overtime == "Yes" else 0

# Create input array
input_data = np.zeros(X.shape[1])
input_data[0] = age
input_data[1] = income
input_data[2] = years
input_data[3] = overtime

# ======================
# PREDICTION
# ======================
if st.sidebar.button("Predict Attrition"):
    prediction = model.predict([input_data])

    if prediction[0] == 1:
        st.error("⚠️ Employee is likely to leave")
    else:
        st.success("✅ Employee is likely to stay")

# ======================
# FEATURE IMPORTANCE
# ======================
st.subheader("📊 Feature Importance")

importances = model.feature_importances_
feat_names = X.columns

imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
imp_df = imp_df.sort_values(by="Importance", ascending=False).head(10)

fig, ax = plt.subplots()
ax.barh(imp_df["Feature"], imp_df["Importance"])
ax.invert_yaxis()
st.pyplot(fig)

# ======================
# BUSINESS INSIGHTS
# ======================
st.subheader("💡 Key Insights")

st.write("""
- Employees working overtime are more likely to leave  
- Lower income employees show higher attrition  
- Certain job roles have higher turnover  
""")
