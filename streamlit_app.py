import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import io

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title(f"Матрица ошибок: {model_name}")
    st.pyplot(fig)

    st.subheader(f"Метрики модели {model_name}")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

def analyze_classification_results(knc, log_reg, dtc, X_test, y_test):
    evaluate_model(knc, X_test, y_test, "K-Nearest Neighbors")
    evaluate_model(log_reg, X_test, y_test, "Logistic Regression")
    evaluate_model(dtc, X_test, y_test, "Decision Tree")


def load_data(url):
    data = pd.read_csv(url, header=None)
    data.columns = [f"A{i}" for i in range(1, 17)]
    return data

def preprocess_features(data):
    data = data.drop(columns=["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"])
    data["A16"] = data["A16"].apply(lambda x: 1 if x == "+" else 0)
    data.replace("?", np.nan, inplace=True)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in data.columns:
        if col != "A16":  
            data[col] = data.groupby("A16")[col].transform(lambda x: x.fillna(x.mean()))
    for col in ["A11", "A14", "A15", "A16"]:
        data[col] = data[col].astype(int)

    return data

    
def feature_selection(data):
    correlation_with_target = data.drop(columns=["A16"]).corrwith(data["A16"]).sort_values(ascending=False)
    st.subheader("🔹 Корреляция признаков с таргетом (A16)")
    st.write(correlation_with_target)
    unique_values_count = data.nunique()
    st.subheader("🔹 Количество уникальных значений в каждом признаке")
    st.write(unique_values_count)
    significant_features = unique_values_count[unique_values_count > 10].index.tolist()
    significant_features_with_correlation = correlation_with_target[significant_features].sort_values(ascending=False)
    st.subheader("🔹 Три наиболее значимые признаки с более чем 10 уникальными значениями")
    st.write(significant_features_with_correlation.head(3))
    return significant_features_with_correlation.head(3).index.tolist()

def plot_3d_graph(data):
    st.subheader("Выбор признаков для 3D-графика")
    
    available_features = list(data.columns)
    available_features.remove("A16") 
    feature_x = st.selectbox("Выберите первый признак (X-ось):", available_features, index=available_features.index("A11"))
    feature_y = st.selectbox("Выберите второй признак (Y-ось):", available_features, index=available_features.index("A8"))
    feature_z = st.selectbox("Выберите третий признак (Z-ось):", available_features, index=available_features.index("A3"))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[feature_x], data[feature_y], data[feature_z], c=data['A16'], marker='o', cmap='viridis')
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_zlabel(feature_z)
    ax.set_title(f'3D-график данных: {feature_x}, {feature_y}, {feature_z}')
    
    fig.colorbar(scatter, ax=ax, label='Класс (A16)')
    st.pyplot(fig)

    
def classification_models(data):
    X = data.drop(columns=["A16"])
    y = data["A16"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
    scaler_option = st.selectbox("Выберите метод нормализации:", ["StandardScaler", "MinMaxScaler"])
    scaler = StandardScaler() if scaler_option == "StandardScaler" else MinMaxScaler()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    st.write("По умолчанию стоят те гиперпарамитры которые вы указали в индивиальных задач")
    kn_neighbors = st.slider("Число соседей (k) для KNN:", min_value=1, max_value=15, value=3)
    log_max_iter = st.slider("Макс. итераций для логистической регрессии:", min_value=100, max_value=1000, value=565, step=50)
    dt_max_depth = st.slider("Макс. глубина дерева решений:", min_value=1, max_value=20, value=5)

    knc = KNeighborsClassifier(n_neighbors=kn_neighbors)
    log_reg = LogisticRegression(max_iter=log_max_iter)
    dtc = DecisionTreeClassifier(max_depth=dt_max_depth)

    X_train = np.array(X_train)[:, :2]
    X_test = np.array(X_test)[:, :2]
    y_train = np.array(y_train)

    knc.fit(X_train, y_train)
    log_reg.fit(X_train, y_train)
    dtc.fit(X_train, y_train)

    st.subheader("🔹 Модели классификации обучены")
    st.write(f"Выбранная нормализация: {scaler_option}")
    st.write(f"KNN (k={kn_neighbors}), Logistic Regression (max_iter={log_max_iter}), Decision Tree (max_depth={dt_max_depth}) успешно обучены.")

    return knc, log_reg, dtc, X_train, X_test, y_train, y_test


def plot_decision_boundaries(X_train, y_train, knc, log_reg, dtc):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    classifiers = [(knc, "K-Nearest Neighbors"), (log_reg, "Logistic Regression"), (dtc, "Decision Tree")]
    for idx, (clf, title) in enumerate(classifiers):
        plt.sca(axes[idx])  
        plot_decision_regions(X_train, y_train, clf=clf, legend=2)
        plt.xlabel("A2")
        plt.ylabel("A3")
        plt.title(title)
    plt.suptitle("граница решений для каждого классификатора ", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(plt)


def plot_roc_curves(knc, log_reg, dtc, X_test, y_test):
    plt.clf()
    y_score_knc = knc.predict_proba(X_test)[:, 1]
    y_score_log_reg = log_reg.predict_proba(X_test)[:, 1]
    y_score_dtc = dtc.predict_proba(X_test)[:, 1]
    fpr_knc, tpr_knc, _ = roc_curve(y_test, y_score_knc)
    fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_score_log_reg)
    fpr_dtc, tpr_dtc, _ = roc_curve(y_test, y_score_dtc)
    auc_knc = auc(fpr_knc, tpr_knc)
    auc_log_reg = auc(fpr_log_reg, tpr_log_reg)
    auc_dtc = auc(fpr_dtc, tpr_dtc)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_knc, tpr_knc, label=f'KNN (AUC = {auc_knc:.2f})', linestyle='-', color='blue')
    plt.plot(fpr_log_reg, tpr_log_reg, label=f'Logistic Regression (AUC = {auc_log_reg:.2f})', linestyle='-', color='orange')
    plt.plot(fpr_dtc, tpr_dtc, label=f'Decision Tree (AUC = {auc_dtc:.2f})', linestyle='-', color='green')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC-кривые')
    plt.legend()
    plt.grid()
    st.pyplot(plt)

def evaluate_models(knc, log_reg, dtc, X_train, X_test, y_train, y_test):
    auc_train_knc = roc_auc_score(y_train, knc.predict_proba(X_train)[:, 1])
    auc_test_knc = roc_auc_score(y_test, knc.predict_proba(X_test)[:, 1])
    auc_train_log_reg = roc_auc_score(y_train, log_reg.predict_proba(X_train)[:, 1])
    auc_test_log_reg = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
    auc_train_dtc = roc_auc_score(y_train, dtc.predict_proba(X_train)[:, 1])
    auc_test_dtc = roc_auc_score(y_test, dtc.predict_proba(X_test)[:, 1])
    results = {"Модель": ["KNN", "Logistic Regression", "Decision Tree"],
                "AUC (Train)": [auc_train_knc, auc_train_log_reg, auc_train_dtc],
                "AUC (Test)": [auc_test_knc, auc_test_log_reg, auc_test_dtc]}

    st.write("**Сравнение моделей по AUC**")
    st.dataframe(results)
    
def evaluate_model(model, X_test, y_test, model_name):
    # Предсказания модели
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title(f"Матрица ошибок: {model_name}")
    st.pyplot(fig)
    st.subheader(f"Метрики модели {model_name}")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())
def analyze_classification_results(knc, log_reg, dtc, X_test, y_test):
    evaluate_model(knc, X_test, y_test, "K-Nearest Neighbors")
    evaluate_model(log_reg, X_test, y_test, "Logistic Regression")
    evaluate_model(dtc, X_test, y_test, "Decision Tree")

def visualize_feature_distributions(data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    features = ["A11", "A8", "A3"]
    for i, feature in enumerate(features):
        sns.histplot(data, x=feature, hue="A16", element="step", bins=20, ax=axes[i], palette="viridis")
        axes[i].set_title(f"Распределение {feature} по классам")
    plt.tight_layout()
    st.pyplot(fig)

def visualize_pairplot(data):
    selected_features = ["A11", "A8", "A3", "A16"]
    pairplot_fig = sns.pairplot(data[selected_features], hue="A16", palette="viridis")
    st.pyplot(pairplot_fig)

def visualize_correlation_matrix(data):
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    plt.title("Матрица корреляций признаков")
    st.pyplot(fig)

def visualize_data(data):
    visualize_feature_distributions(data)
    visualize_pairplot(data)
    visualize_correlation_matrix(data)



def main():
    st.title("📊 Анализ набора данных из репозитория UCI")
    st.subheader("🔹 Шаг 1: Загрузка данных")
    url = st.text_input("Введите URL для загрузки данных:", 
                       "https://raw.githubusercontent.com/AMIROLIMI/ML_DS_Jun/master/HW4-crx.data")
    if url:
        data = load_data(url)

        num_rows = st.slider("Выберите количество строк для отображения:", 
                             min_value=4, max_value=len(data), value=5)
        
        st.subheader("Предварительный просмотр данных")
        st.dataframe(data.head(num_rows))

        st.subheader("🔹 Шаг 2: Обработка меток классов")
        st.subheader("📉 Количество пропущенных значений до обработки данных")
        st.write(data.isna().sum())

        buffer = io.StringIO()
        data.info(buf=buffer)
        info_str = buffer.getvalue()

        st.subheader("🔢 Информация о данных после обработки")
        st.text(info_str)
        st.subheader("Уникальные значения в столбце A16 до обработки")
        st.write(data["A16"].unique())
        st.markdown(""" 
        В этих данных нет пропущенных значений, но есть неправильные значения как "?". 
        Эти значения будут удалены и заменены средними значениями. А классов 2, по этому можно ничего не делать.
        """)
        
        st.markdown(""" 
        Мы заменили символ "+" на 1 и символ "-" на 0 в столбце A16.
        """)
        st.subheader("Обработанные метки классов")
        st.dataframe(data.head(num_rows))
        st.subheader("🔹 Шаг 3: Предобработка признаков")
        st.markdown(""" 
        ### Сначала удалим реально категориальные признаки, чтобы было удобно:
        Поля, которые удалятся: ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]
        
        После этого заменим все неправильные значения, такие как "?" на средние значения.
        """)
        
        processed_data = preprocess_features(data)
        st.subheader("Обработанные данные")
        st.dataframe(processed_data.head(num_rows))
        buffer = io.StringIO()
        processed_data.info(buf=buffer)
        info_str = buffer.getvalue()
        st.subheader("🔢 Информация о данных после обработки")
        st.text(info_str)
        st.subheader("📉 Количество пропущенных значений после обработки данных")
        st.write(processed_data.isna().sum())

        csv = processed_data.to_csv(index=False)
        st.download_button("Скачать обработанные данные", csv, "processed_data.csv", "text/csv")

        # Шаг 4:
        st.subheader("🔹 Шаг 4: Отбор признаков")
        significant_features = feature_selection(processed_data)
        st.subheader("🔹 Три наиболее значимые признаки:")
        st.write(significant_features)
        st.subheader("🔹 Шаг 5: Визуализация данных")
        plot_3d_graph(processed_data)
        st.subheader("🔹 Шаг 6: Классификация")
        knc, log_reg, dtc, X_train, X_test, y_train, y_test = classification_models(processed_data)
        evaluate_models(knc, log_reg, dtc, X_train, X_test, y_train, y_test)
        st.subheader("🔹 Шаг 7: Визуализация границ решений")
        #plot_decision_boundaries(X_train, y_train, knc, log_reg, dtc)
        st.subheader("🔹 Шаг 8: ROC-кривые") 
        plot_roc_curves(knc, log_reg, dtc, X_test, y_test)
        st.subheader("🔹 9. Оценка качества классификации")
        evaluate_models(knc, log_reg, dtc, X_train, X_test, y_train, y_test)
        st.write("Видно что у маделей KNN и Decission Tree переобучение так как у них на трейне высокий показатель а на тесте низкий. А у модели Logistic Regression такого нет. по этому будем считать что самый хорошый модель это - Logistic Regression")
        st.subheader("🔹 10. Дополнительно")
        analyze_classification_results(knc, log_reg, dtc, X_test, y_test)
        visualize_data(processed_data)

if __name__ == "__main__":
    main()
