import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file, header=None)
    data.columns = [f"A{i}" for i in range(1, 17)]
    return data

def preprocess_data(data):
    data["A16"] = data["A16"].apply(lambda x: 1 if x == "+" else 0)
    data = data.drop(columns=["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"])
    
    data.replace("?", np.nan, inplace=True)
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    for col in ["A11", "A14", "A15", "A16"]:
        data[col] = data[col].astype(int)
    
    return data

def main():
    st.title("📊 Анализ набора данных UCI")
    st.markdown("Загрузите файл с данными для анализа.")
    
    uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.subheader("Предварительный просмотр данных")
        st.dataframe(data.head())
        
        if st.button("🔄 Обработать данные"):
            processed_data = preprocess_data(data)
            st.subheader("Обработанные данные")
            st.dataframe(processed_data.head())
            
            st.subheader("🔢 Информация о данных")
            st.text(processed_data.info())
            
            st.subheader("📉 Количество пропущенных значений")
            st.write(processed_data.isna().sum())

if __name__ == "__main__":
    main()
