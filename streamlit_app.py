import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_data(url):
    data = pd.read_csv(url, header=None)
    data.columns = [f"A{i}" for i in range(1, 17)]
    return data

def preprocess_target(data):
    data = data.dropna(subset=["A16"])
    class_counts = data["A16"].value_counts()
    if len(class_counts) > 2:
        majority_class = class_counts.idxmax()
        data["A16"] = data["A16"].apply(lambda x: 1 if x == majority_class else 0)
    return data

def preprocess_features(data):
    data = data.drop(columns=["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"])
    data.replace("?", np.nan, inplace=True)
    
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    for col in ["A11", "A14", "A15", "A16"]:
        data[col] = data[col].astype(int)
    
    return data

def main():
    st.title("📊 Анализ набора данных из репозитория UCI")
    st.subheader("🔹 Шаг 1: 🔹 1. Загрузка данных")
    st.markdown("""
    - Загрузите набор данных из репозитория UCI, включая столбец с метками классов, указанный в индивидуальном задании.
    """)
    st.markdown("Загрузите данные, выполните их обработку и предобработку.")

    url = st.text_input("Введите URL для загрузки данных:", 
                       "https://raw.githubusercontent.com/AMIROLIMI/ML_DS_Jun/master/HW4-crx.data")
    
    if url:

        data = load_data(url)

        num_rows = st.slider("Выберите количество строк для отображения:", 
                             min_value=4, max_value=len(data), value=5)
        
        st.subheader("Предварительный просмотр данных")
        st.dataframe(data.head(num_rows))

        st.subheader("🔹 Шаг 2: Обработка меток классов")
        st.markdown("""
        - 🛠 Если есть пропущенные значения в метках классов, удалите соответствующие записи.
        - 🔄 Если классов больше двух, объедините их так, чтобы получилась бинарная классификация с примерно равным количеством примеров.
        - ⚖️ Если один класс преобладает, объедините все остальные в один. Всего должно остаться 2 класса в таргете.
        """)
        if st.button("🔄 Обработать метки классов"):
            processed_data = preprocess_target(data)
            st.subheader("Обработанные метки классов")
            st.dataframe(processed_data.head(num_rows))
        
        st.subheader("🔹 Шаг 3: Предобработка признаков")
        st.markdown("""
        - 🔢 Преобразуйте числовые признаки, если они были неправильно распознаны.
        - 🗑 Удалите категориальные признаки (текстовые значения).
        - 📉 Заполните пропущенные значения средними значениями отдельно для положительного и отрицательного классов.
        """)
        
        st.subheader("📉 Количество пропущенных значений до обработки данных")
        st.write(data.isna().sum())
        
        impute_strategy = st.selectbox("Выберите стратегию для заполнения пропущенных значений:", 
                                      ["mean", "median", "most_frequent"])
        
        if st.button("🔄 Обработать данные"):
            processed_data = preprocess_features(processed_data)
            st.subheader("Обработанные данные")
            st.dataframe(processed_data.head(num_rows))
            
            st.subheader("🔢 Информация о данных")
            st.text(processed_data.info())
            
            st.subheader("📉 Количество пропущенных значений после обработки данных")
            st.write(processed_data.isna().sum())
            
            csv = processed_data.to_csv(index=False)
            st.download_button("Скачать обработанные данные", csv, "processed_data.csv", "text/csv")

if __name__ == "__main__":
    main()
