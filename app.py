import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and target names
try:
    model, target_names = joblib.load("model.pkl")
except Exception as e:
    st.error("❌ Failed to load model. Make sure iris_model.pkl exists.")
    st.stop()

st.title("🌼 Iris Flower Prediction App")

# Input features
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    # Create input DataFrame
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=['sepal length (cm)', 'sepal width (cm)',
                                       'petal length (cm)', 'petal width (cm)'])

    st.write("📦 Input Data:")
    st.write(input_data)

    # Make prediction
    prediction = model.predict(input_data)[0]
    try:
        proba = model.predict_proba(input_data)[0]
    except Exception as e:
        st.error(f"❌ Error getting prediction probabilities: {e}")
        st.stop()

    st.subheader("🌸 Prediction Result")
    st.write(f"**Predicted Species:** {target_names[prediction]}")
    st.write(f"**Prediction Confidence:** {max(proba) * 100:.2f}%")

    # Plot class probabilities
    st.subheader("📊 Class Probability Distribution")
    fig, ax = plt.subplots()
    sns.barplot(x=target_names, y=proba, palette="pastel", ax=ax)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
