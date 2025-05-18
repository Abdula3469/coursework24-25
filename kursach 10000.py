import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                             VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.datasets import fashion_mnist

# В данном коде я наставил большущее количество принтов, дабы отслеживать какая сейчас часть делается, ведь для выполнения кода этого, в прошлый раз, мне понадобилось 4 часа
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ["Футболка", "Брюки", "Пуловер", "Платье", "Пальто",
               "Сандаль", "Рубашка", "Кроссовок", "Сумка", "Ботинок"]

print("=== 1. Описание набора данных и задачи ===")
print(f"Размер тренировочной выборки: {train_images.shape[0]} изображений")
print(f"Размер тестовой выборки: {test_images.shape[0]} изображений")
print(f"Разрешение изображений: {train_images.shape[1]}x{train_images.shape[2]} пикселей")
print("Классы:", class_names)


print("\n=== 2. Предварительный анализ ===")
print("Пример изображения:")
plt.figure(figsize=(3,3))
plt.imshow(train_images[0], cmap='gray')
plt.title(f"Метка: {class_names[train_labels[0]]}")
plt.axis('off')
plt.show()


print(f"Отсутствующие значения в train: {np.isnan(train_images).sum()}")
print(f"Отсутствующие значения в test: {np.isnan(test_images).sum()}")


X_train = train_images.reshape((train_images.shape[0], -1)) / 255.0
X_test = test_images.reshape((test_images.shape[0], -1)) / 255.0
y_train = train_labels
y_test = test_labels


print("\n=== 4. Описательный анализ ===")

plt.figure(figsize=(10,5))
sns.countplot(x=train_labels)
plt.xticks(ticks=range(10), labels=class_names, rotation=45)
plt.title("Распределение классов в тренировочной выборке")
plt.show()


plt.figure(figsize=(15,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train[y_train==i].mean(axis=0).reshape(28,28), cmap='gray')
    plt.title(class_names[i])
    plt.axis('off')
plt.suptitle("Средние изображения для каждого класса")
plt.show()


print("\n=== 5. Кластеризация ===")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train[:1000])  

kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X_pca)

plt.figure(figsize=(10,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='tab10', alpha=0.6)
plt.title("K-means кластеризация (PCA 2D проекция)")
plt.colorbar(scatter)
plt.show()


print("\n=== 6. Разделение данных ===")
print("Используется стандартное разделение Fashion MNIST:")
print(f"Тренировочная выборка: {X_train.shape[0]} образцов")
print(f"Тестовая выборка: {X_test.shape[0]} образцов")


print("\n=== 7. Обучение моделей ===")
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=50,           
        learning_rate=0.2,         
        max_depth=3,               
        subsample=0.5,            
        validation_fraction=0.1,   
        n_iter_no_change=5,       
        random_state=42
    ),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
}

results = {}
for name, model in models.items():
    print(f"Обучение {name}...")
    start = time()

    
    if name == "Gradient Boosting":
        from tqdm import tqdm
        pbar = tqdm(total=model.n_estimators, desc="Обучение GB", unit="tree")

       
        def callback(env):
            pbar.update(1)
            if env.iteration == env.end_iteration - 1:
                pbar.close()

        model.set_params(init='zero', warm_start=False)
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    train_time = time() - start

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        'model': model,
        'accuracy': acc,
        'f1_score': f1,
        'train_time': train_time
    }
    print(f"{name}: Точность={acc:.3f}, F1={f1:.3f}, Время={train_time:.1f}с")


results_df = pd.DataFrame.from_dict(results, orient='index')
print("\nСравнение моделей:")
print(results_df.sort_values('accuracy', ascending=False))


print("\n=== 8. Ансамблевые методы ===")

top_models = [k for k in sorted(results, key=lambda x: results[x]['accuracy'], reverse=True)[:3]]


top_models = ["Random Forest", "Gradient Boosting"]  # Лучшие по точности
voting = VotingClassifier(
    estimators=[(name, results[name]['model']) for name in top_models],
    voting='soft'
)
stacking = StackingClassifier(
    estimators=[(name, results[name]['model']) for name in ["Random Forest", "Gradient Boosting", "SVM"]],
    final_estimator=RandomForestClassifier(n_estimators=50)  # Упрощенная модель
)


for ensemble in [voting, stacking]:
    name = type(ensemble).__name__
    print(f"Обучение {name}...")
    start = time()
    ensemble.fit(X_train, y_train)
    train_time = time() - start

    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        'model': ensemble,
        'accuracy': acc,
        'f1_score': f1,
        'train_time': train_time
    }
    print(f"{name}: Точность={acc:.3f}, F1={f1:.3f}, Время={train_time:.1f}с")


print("\n=== 9. Улучшение моделей ===")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train[:5000], y_train[:5000])  # Уменьшаем выборку для скорости

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nЛучшие параметры: {grid_search.best_params_}")
print(f"Точность улучшенной модели: {acc:.3f} (было {results['Random Forest']['accuracy']:.3f})")


print("\n=== 10. Визуализация результатов ===")

results_df = pd.DataFrame.from_dict(results, orient='index')
results_df = results_df.sort_values('accuracy', ascending=False)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.barplot(x=results_df.index, y=results_df['accuracy'])
plt.title('Точность моделей')
plt.xticks(rotation=45)
plt.ylim(0.7, 0.9)

plt.subplot(1,2,2)
sns.barplot(x=results_df.index, y=results_df['train_time'])
plt.title('Время обучения')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


best_model_name = results_df.iloc[0].name
best_model = results[best_model_name]['model']
y_pred = best_model.predict(X_test)

plt.figure(figsize=(12,10))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Матрица ошибок ({best_model_name})\nТочность: {results[best_model_name]["accuracy"]:.2%}')
plt.xlabel('Предсказано')
plt.ylabel('Истинное значение')
plt.show()


errors = np.where(y_pred != y_test)[0]
np.random.shuffle(errors)
plt.figure(figsize=(15,3))
for i, idx in enumerate(errors[:5]):
    plt.subplot(1,5,i+1)
    plt.imshow(test_images[idx], cmap='gray')
    true = class_names[y_test[idx]]
    pred = class_names[y_pred[idx]]
    plt.title(f"True: {true}\nPred: {pred}", fontsize=9)
    plt.axis('off')
plt.suptitle('Примеры ошибочных классификаций', y=1.05)
plt.tight_layout()
plt.show()


print("\n=== Выводы ===")
print("1. Лучшая модель:", best_model_name)
print(f"2. Точность лучшей модели: {results[best_model_name]['accuracy']:.2%}")
print("3. Ансамблевые методы показали улучшение точности на",
      f"{results[best_model_name]['accuracy'] - results[top_models[0]]['accuracy']:.2%}")
print("4. Основные ошибки классификации связаны с визуально похожими классами:")
print("   - Рубашки/Футболки/Пуловеры")
print("   - Сандалии/Кроссовки/Ботинки")
