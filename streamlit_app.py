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
    st.subheader("\U0001F4CA Анализ набора данных из репозитория UCI")
    st.subheader("\U0001F4A1 Шаг 1: Загрузка данных")
    st.markdown("""
    - Загрузите набор данных из репозитория UCI, включая столбец с метками классов, указанный в индивидуальном задании.
    """)
    data = pd.read_csv(url, header=None)
    data.columns = [f"A{i}" for i in range(1, 17)]
    return data

def preprocess_features(data):
    st.subheader("\U0001F4A1 Шаг 2: Обработка меток классов")
    st.markdown("""
    - 🛠 Если есть пропущенные значения в метках классов, удалите соответствующие записи.
    - 🔄 Если классов больше двух, объедините их так, чтобы получилась бинарная классификация с примерно равным количеством примеров.
    - ⚖️ Если один класс преобладает, объедините все остальные в один. Всего должно остаться 2 класса в таргете.
    """)
    data["A16"] = data["A16"].apply(lambda x: 1 if x == "+" else 0)
    
    st.subheader("\U0001F4A1 Шаг 3: Предобработка признаков")
    st.markdown("""
    - 🔢 Преобразуйте числовые признаки, если они были неправильно распознаны.
    - 🗑 Удалите категориальные признаки (текстовые значения).
    - 📉 Заполните пропущенные значения средними значениями отдельно для положительного и отрицательного классов.
    """)
    data = data.drop(columns=["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"])
    data.replace("?", np.nan, inplace=True)
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    for col in ["A11", "A14", "A15", "A16"]:
        data[col] = data[col].astype(int)
    return data

def feature_selection(data):
    st.subheader("\U0001F4A1 Шаг 4: Отбор признаков")
    st.markdown("""
    - 🔍 Определите три наиболее значимых признака, содержащих более 10 уникальных значений по корреляции с таргетом.
    """)
    correlation_with_target = data.drop(columns=["A16"]).corrwith(data["A16"]).sort_values(ascending=False)
    st.write(correlation_with_target)
    unique_values_count = data.nunique()
    significant_features = unique_values_count[unique_values_count > 10].index.tolist()
    significant_features_with_correlation = correlation_with_target[significant_features].sort_values(ascending=False)
    return significant_features_with_correlation.head(3).index.tolist()

def plot_3d_graph(data):
    st.subheader("\U0001F4A1 Шаг 5: Визуализация данных")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data['A11'], data['A8'], data['A3'], c=data['A16'], marker='o', cmap='viridis')
    ax.set_xlabel('A11')
    ax.set_ylabel('A8')
    ax.set_zlabel('A3')
    ax.set_title('3D-график данных')
    fig.colorbar(scatter, ax=ax, label='Класс (A16)')
    st.pyplot(fig)

def classification_models(data):
    st.subheader("\U0001F4A1 Шаг 6: Обучение моделей классификации")
    X = data.drop(columns=["A16"])
    y = data["A16"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    knc = KNeighborsClassifier(n_neighbors=3)
    log_reg = LogisticRegression(max_iter=565)
    dtc = DecisionTreeClassifier(max_depth=5)
    knc.fit(X_train, y_train)
    log_reg.fit(X_train, y_train)
    dtc.fit(X_train, y_train)
    return knc, log_reg, dtc, X_train, X_test, y_train, y_test

def plot_decision_boundaries(X_train, y_train):
    st.subheader("\U0001F4A1 Шаг 7: Границы решений")
    X_train_np = np.array(X_train)[:, :2]
    y_train_np = np.array(y_train)
    knc = KNeighborsClassifier()
    log_reg = LogisticRegression()
    dtc = DecisionTreeClassifier()
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
    st.pyplot()

def plot_roc_curves(knc, log_reg, dtc, X_test, y_test):
    st.subheader("\U0001F4A1 Шаг 8: ROC-кривые")
    y_score_knc = knc.predict_proba(X_test)[:, 1]
    y_score_log_reg = log_reg.predict_proba(X_test)[:, 1]
    y_score_dtc = dtc.predict_proba(X_test)[:, 1]
    fpr_knc, tpr_knc, _ = roc_curve(y_test, y_score_knc)
    fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_score_log_reg)
    fpr_dtc, tpr_dtc, _ = roc_curve(y_test, y_score_dtc)
    plt.plot(fpr_knc, tpr_knc, label='KNN')
    plt.plot(fpr_log_reg, tpr_log_reg, label='Logistic Regression')
    plt.plot(fpr_dtc, tpr_dtc, label='Decision Tree')
    plt.legend()
    st.pyplot()

def main():
    url = st.text_input("Введите URL:", "https://raw.githubusercontent.com/AMIROLIMI/ML_DS_Jun/master/HW4-crx.data")
    if url:
        data = load_data(url)
        processed_data = preprocess_features(data)
        significant_features = feature_selection(processed_data)
        plot_3d_graph(processed_data)
        knc, log_reg, dtc, X_train, X_test, y_train, y_test = classification_models(processed_data)
        plot_decision_boundaries(X_train, y_train)
        plot_roc_curves(knc, log_reg, dtc, X_test, y_test)

if __name__ == "__main__":
    main()
