import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_data():
    # 📊 Загрузка данных из репозитория UCI
    url = "https://raw.githubusercontent.com/AMIROLIMI/ML_DS_Jun/master/HW4-crx.data"
    data = pd.read_csv(url, header=None)
    data.columns = [f"A{i}" for i in range(1, 17)]
    return data

def preprocess_data(data):
    # 🛠 Обработка меток классов
    data["A16"] = data["A16"].apply(lambda x: 1 if x == "+" else 0)
    
    # 🔢 Преобразование и удаление категориальных признаков
    data = data.drop(columns=["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"])
    
    # 📉 Заполнение пропущенных значений
    data.replace("?", np.nan, inplace=True)
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # Преобразование типов данных
    for col in ["A11", "A14", "A15", "A16"]:
        data[col] = data[col].astype(int)
    
    return data

def main():
    st.title("📊 Анализ набора данных UCI")
    st.markdown("Загрузка данных и их предобработка.")
    
    data = load_data()
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
