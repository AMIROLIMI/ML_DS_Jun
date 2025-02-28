import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_data(url):
    data = pd.read_csv(url, header=None)
    data.columns = [f"A{i}" for i in range(1, 17)]
    return data


def preprocess_features(data):
    data = data.drop(columns=["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"])
    data["A16"] = data["A16"].apply(lambda x: 1 if x == "+" else 0)
    data.replace("?", np.nan, inplace=True)
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    for col in ["A11", "A14", "A15", "A16"]:
        data[col] = data[col].astype(int)
    return data
    
def feature_selection(data):
    # 1. Корреляция с таргетом (A16)
    correlation_with_target = data.drop(columns=["A16"]).corrwith(data["A16"]).sort_values(ascending=False)
    st.subheader("🔹 Корреляция признаков с таргетом (A16)")
    st.write(correlation_with_target)

    # 2. Количество уникальных значений в каждом признаке
    unique_values_count = data.nunique()
    st.subheader("🔹 Количество уникальных значений в каждом признаке")
    st.write(unique_values_count)

    # 3. Отбор признаков с более чем 10 уникальными значениями
    significant_features = unique_values_count[unique_values_count > 10].index.tolist()
    significant_features_with_correlation = correlation_with_target[significant_features].sort_values(ascending=False)
    
    st.subheader("🔹 Три наиболее значимые признаки с более чем 10 уникальными значениями")
    st.write(significant_features_with_correlation.head(3))

    return significant_features_with_correlation.head(3).index.tolist()

def main():
    st.title("📊 Анализ набора данных из репозитория UCI")
    st.subheader("🔹 Шаг 1: Загрузка данных")
    st.markdown("""
    - Загрузите набор данных из репозитория UCI, включая столбец с метками классов, указанный в индивидуальном задании.
    """)
    
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
        
        st.subheader("📉 Количество пропущенных значений до обработки данных")
        st.write(data.isna().sum())

        st.subheader("🔢 Информация о данных до обработки")
        st.text(data.info())

        st.subheader("Уникальные значения в столбце A16 до обработки")
        st.write(data["A16"].unique())
        
        # Обработка меток классов
        st.markdown(""" 
        В этих данных нет пропущенных значений, но есть неправильные значения как "?". 
        Эти значения будут удалены и заменены средними значениями. А классов 2, по этому можно ничего не делать.
        """)
        data["A16"] = data["A16"].apply(lambda x: 1 if x == "+" else 0)
        
        st.markdown(""" 
        Мы заменили символ "+" на 1 и символ "-" на 0 в столбце A16.
        """)
        
        st.subheader("Обработанные метки классов")
        st.dataframe(data.head(num_rows))

        st.subheader("🔹 Шаг 3: Предобработка признаков")
        st.markdown("""
        - 🔢 Преобразуйте числовые признаки, если они были неправильно распознаны.
        - 🗑 Удалите категориальные признаки (текстовые значения).
        - 📉 Заполните пропущенные значения средними значениями отдельно для положительного и отрицательного классов.
        """)
        st.markdown(""" 
        ### Сначала удалим реально категориальные признаки, чтобы было удобно:
        Поля, которые удалятся: ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]
        
        После этого заменим все неправильные значения, такие как "?" на средние значения.
        """)
        
        processed_data = preprocess_features(data)
        st.subheader("Обработанные данные")
        st.dataframe(processed_data.head(num_rows))
        
        st.subheader("🔢 Информация о данных после обработки")
        st.text(processed_data.info())
        
        st.subheader("📉 Количество пропущенных значений после обработки данных")
        st.write(processed_data.isna().sum())

        csv = processed_data.to_csv(index=False)
        st.download_button("Скачать обработанные данные", csv, "processed_data.csv", "text/csv")

        # Шаг 4: Отбор признаков
        st.subheader("🔹 Шаг 4: Отбор признаков")
        st.markdown("""
        - 🔍 Определите три наиболее значимых признака, содержащих более 10 уникальных значений по корреляции с таргетом.
        - Мы вычислим корреляцию признаков с целевой переменной (A16) и выберем три наиболее значимых.
        - Признаки с более чем 10 уникальными значениями будут отобраны по наибольшей корреляции с целевой переменной.
        """)
        
        st.dataframe(data.head(num_rows))

        # Вычисление значимых признаков
        significant_features = feature_selection(processed_data)
        st.subheader("🔹 Три наиболее значимые признаки:")
        st.write(significant_features)

if __name__ == "__main__":
    main()
