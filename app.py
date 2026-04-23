import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Title
st.title("Employee Attrition Prediction App")

# Load data
df = pd.read_csv("HR-Employee-Attrition.csv")

# Show dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Encode target
df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})

# Encode categorical
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split data
X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Plot Attrition
st.subheader("Attrition Count")
fig, ax = plt.subplots()
sns.countplot(x='Attrition', data=df, ax=ax)
st.pyplot(fig)

# Prediction section
st.subheader("Predict Employee Attrition")

age = st.slider("Age", 18, 60, 30)
income = st.number_input("Monthly Income", 1000, 20000, 5000)

if st.button("Predict"):
    input_data = [[age] + [0]*(X.shape[1]-1)]
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Employee likely to leave")
    else:
        st.success("Employee likely to stay")