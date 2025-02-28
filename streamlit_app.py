import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(url):
    # 📊 Загрузка данных из URL
    data = pd.read_csv(url, header=None)
    data.columns = [f"A{i}" for i in range(1, 17)]
    return data

def preprocess_data(data, impute_strategy='mean'):
    # 🛠 Обработка меток классов
    data["A16"] = data["A16"].apply(lambda x: 1 if x == "+" else 0)
    
    # 🔢 Преобразование и удаление категориальных признаков
    data = data.drop(columns=["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"])
    
    # 📉 Заполнение пропущенных значений
    data.replace("?", np.nan, inplace=True)
    imputer = SimpleImputer(strategy=impute_strategy)
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # Преобразование типов данных
    for col in ["A11", "A14", "A15", "A16"]:
        data[col] = data[col].astype(int)
    
    return data

def plot_data(data):
    # 📈 Визуализация данных
    st.subheader("📊 Графики распределений")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sns.histplot(data["A11"], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Распределение A11")
    
    sns.histplot(data["A14"], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Распределение A14")
    
    sns.histplot(data["A15"], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Распределение A15")
    
    sns.countplot(x="A16", data=data, ax=axes[1, 1])
    axes[1, 1].set_title("Распределение метки A16")
    
    st.pyplot(fig)

def main():
    st.title("📊 Анализ набора данных UCI")
    st.markdown("Загрузка данных и их предобработка.")
    
    # Выбор URL для загрузки данных
    url = st.text_input("Введите URL для загрузки данных:", 
                       "https://raw.githubusercontent.com/AMIROLIMI/ML_DS_Jun/master/HW4-crx.data")
    
    if url:
        data = load_data(url)
        
        # Слайдер для выбора количества строк
        num_rows = st.slider("Выберите количество строк для отображения:", 
                             min_value=1, max_value=len(data), value=5)
        
        st.subheader("Предварительный просмотр данных")
        st.dataframe(data.head(num_rows))
        
        # Выбор стратегии обработки пропущенных значений
        impute_strategy = st.selectbox("Выберите стратегию для заполнения пропущенных значений:", 
                                      ["mean", "median", "most_frequent"])
        
        if st.button("🔄 Обработать данные"):
            processed_data = preprocess_data(data, impute_strategy)
            st.subheader("Обработанные данные")
            st.dataframe(processed_data.head(num_rows))
            
            st.subheader("🔢 Информация о данных")
            st.text(processed_data.info())
            
            st.subheader("📉 Количество пропущенных значений")
            st.write(processed_data.isna().sum())
            
            # Графики
            plot_data(processed_data)
            
            # Кнопка для скачивания обработанных данных
            csv = processed_data.to_csv(index=False)
            st.download_button("Скачать обработанные данные", csv, "processed_data.csv", "text/csv")

if __name__ == "__main__":
    main()
