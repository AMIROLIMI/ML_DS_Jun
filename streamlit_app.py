import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from mlxtend.plotting import plot_decision_regions



def load_data(url):
    data = pd.read_csv(url, header=None)
    data.columns = [f"A{i}" for i in range(1, 17)]
    return data


def preprocess_features(data):
    data = data.drop(columns=["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"])
    data["A16"] = data["A16"].apply(lambda x: 1 if x == "+" else 0)
    st.dataframe(data["A16"].head(690))
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
    st.dataframe(data["A16"].head(690))

    return significant_features_with_correlation.head(3).index.tolist()

def plot_3d_graph(data):
    st.subheader("🔹 Шаг 5: Визуализация данных")
    st.dataframe(data["A16"].head(690))
    st.markdown("""
    - 🎨 Постройте 3D-график точек данных, используя разные цвета и маркеры для классов.
    - 🏷 Подпишите оси названиями признаков.
    - 🖼 Добавьте заголовок с названием набора данных и легенду.
    """)
    
    # Создаем 3D график
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(data['A11'], data['A8'], data['A3'], c=data['A16'], marker='o', cmap='viridis')
    
    # Подпись осей
    ax.set_xlabel('A11')
    ax.set_ylabel('A8')
    ax.set_zlabel('A3')
    
    # Заголовок
    ax.set_title('3D-график данных: A3, A8, A11 (Жёлтый цвет положительный класс, а остальные отрицательный класс.)')
    fig.colorbar(scatter, ax=ax, label='Класс (A16)')
    st.pyplot(fig)
    
def classification_models(data):
    X = data.drop(columns=["A16"])
    y = data["A16"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
    st.dataframe(data["A16"].head(690))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    knc = KNeighborsClassifier(n_neighbors=3)
    log_reg = LogisticRegression(max_iter=565)
    dtc = DecisionTreeClassifier(max_depth = 5)
    st.subheader("🔹 Модели классификации обучены")
    st.write("K-Nearest Neighbors, Logistic Regression, Decision Tree успешно обучены на данных.")
    
    return knc, log_reg, dtc, X_train, X_test, y_train, y_test

def plot_decision_boundaries(X_train, y_train, knc, log_reg, dtc):
    X_train_np = np.array(X_train)[:, :2]
    y_train_np = np.array(y_train)
    
    knc.fit(X_train_np, y_train_np)
    log_reg.fit(X_train_np, y_train_np)
    dtc.fit(X_train_np, y_train_np)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    classifiers = [(knc, "K-Nearest Neighbors"), (log_reg, "Logistic Regression"), (dtc, "Decision Tree")]
    
    for idx, (clf, title) in enumerate(classifiers):
        plt.sca(axes[idx])  
        plot_decision_regions(X_train_np, y_train_np, clf=clf, legend=2)
        plt.xlabel("A2")
        plt.ylabel("A3")
        plt.title(title)
    
    plt.suptitle("граница решений для каждого классификатора ", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    st.pyplot(plt)


def plot_roc_curves(knc, log_reg, dtc, X_test, y_test):
    y_score_knc = knc.predict_proba(X_test)[:, 1]
    y_score_log_reg = log_reg.predict_proba(X_test)[:, 1]
    y_score_dtc = dtc.predict_proba(X_test)[:, 1]
    fpr_knc, tpr_knc, _ = roc_curve(y_test, y_score_knc)
    fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_score_log_reg)
    fpr_dtc, tpr_dtc, _ = roc_curve(y_test, y_score_dtc)
    auc_knc = auc(fpr_knc, tpr_knc)
    auc_log_reg = auc(fpr_log_reg, tpr_log_reg)
    auc_dtc = auc(fpr_dtc, tpr_dtc)
    plt.plot(fpr_knc, tpr_knc, label=f'KNN (AUC = {auc_knc:.2f})', linestyle='-', color='blue')
    plt.plot(fpr_log_reg, tpr_log_reg, label=f'Logistic Regression (AUC = {auc_log_reg:.2f})', linestyle='-', color='orange')
    plt.plot(fpr_dtc, tpr_dtc, label=f'Decision Tree (AUC = {auc_dtc:.2f})', linestyle='-', color='green')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC-кривые')
    plt.legend()
    plt.grid()
    plt.show()
    st.pyplot(plt)

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
        # шаг 5
        plot_3d_graph(processed_data)
        st.subheader("🔹 Шаг 6: Классификация")
        knc, log_reg, dtc, X_train, X_test, y_train, y_test = classification_models(processed_data)
        st.subheader("🔹 Шаг 7: Визуализация границ решений")
        plot_decision_boundaries(X_train, y_train, knc, log_reg, dtc)
        st.subheader("🔹 Шаг 8: ROC-кривые") 
        plot_roc_curves(knc, log_reg, dtc, X_test, y_test)

if __name__ == "__main__":
    main()
