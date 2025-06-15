# Librerías
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from scipy.stats import zscore
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import early_stopping, log_evaluation

# Cargar datos
data = 'data/ObesityDataSet_raw_and_data_sinthetic.xlsx'
df = pd.read_excel(data)

# Punto 7
head = df.head()
head.to_excel('res/tables/head.xlsx', index=False)

# Punto 8
variables_cuantitativas = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
resumen_metricas = df[variables_cuantitativas].describe().loc[['min', '25%', '50%', '75%', 'max', 'mean']]
resumen_metricas.to_excel('res/tables/resumen.xlsx', index=True)

# Punto 9
for var in variables_cuantitativas:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[var], kde=True, bins=30)
    plt.title(f'Distribución de {var}')
    plt.xlabel(var)
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()

# Punto 10
# Matriz de correlación
corr_matrix = df[variables_cuantitativas].corr()
# Mapa de calor
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de correlación entre variables cuantitativas')
plt.show()
# Calcular matriz de correlación de Pearson
pearson_corr = df[variables_cuantitativas].corr(method='pearson')
# Mapa de calor
plt.figure(figsize=(10, 6))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Mapa de calor - Correlación de Pearson entre variables cuantitativas')
plt.tight_layout()
plt.show()
# Relacion entre Obesidad y Peso
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='NObeyesdad', y='Weight', palette='Set2')
plt.xticks(rotation=45)
plt.title('Relación entre niveles de obesidad y peso')
plt.xlabel('Nivel de obesidad')
plt.ylabel('Peso (kg)')
plt.tight_layout()
plt.show()

# Punto 11
# Variables categóricas
variables_categoricas = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
for var in variables_categoricas:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=var, order=df[var].value_counts().index)
    plt.title(f'Distribución de {var}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Punto 12    
# Valores faltantes
print("Valores faltantes por variable:")
print(df.isnull().sum())
# Detección de outliers con boxplots
for var in variables_cuantitativas:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=var)
    plt.title(f'Detección de outliers en {var}')
    plt.tight_layout()
    plt.show()
    
# Punto 13
# Tecnicas comunes para Valores faltantes
# Eliminar observaciones con NA
#df.dropna(inplace=True)
# Imputación para variables cuantitativas
#df['Variable'].fillna(df['Variable'].mean(), inplace=True)
# Imputacion para variables cualitativas
#df['Variable'].fillna(df['Variable'].mode()[0], inplace=True)
# Interpolación
#df.interpolate(method='linear', inplace=True)
# Modelos predictivos, uso de algoritmos como KNN, regresión, árboles de decisión.

# Tecnicas comunes para Outliers
# Metodo IQR, rango intercuartilico
#Q1 = df['Weight'].quantile(0.25)
#Q3 = df['Weight'].quantile(0.75)
#IQR = Q3 - Q1
#outliers = df[(df['Weight'] < Q1 - 1.5 * IQR) | (df['Weight'] > Q3 + 1.5 * IQR)]
# Estandarización (Z-Score)
#from scipy.stats import zscore
#df['z_weight'] = zscore(df['Weight'])
#outliers = df[df['z_weight'].abs() > 3]

# Variables predictoras y objetivo
objetivo = 'NObeyesdad'
X = df.drop(columns=[objetivo])
y = df[objetivo]

# Para variables categoricas
cat_cols = X.select_dtypes(include='object').columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Label para variable objetivo
le_objetivo = LabelEncoder()
y_encoded = le_objetivo.fit_transform(y)

# Dividir dataset en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Modelo LGBMClassifier 
clf = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(le_objetivo.classes_),
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=100
)

# Entrenando el modelo con Callbacks
clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='multi_logloss',
    callbacks=[
        early_stopping(stopping_rounds=10),
        log_evaluation(period=0)
    ]
)

# Predicción
y_pred_labels = clf.predict(X_test)

# Evaluación
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_labels, objetivo_names=le_objetivo.classes_))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_labels))

# Importancia de variables
feature_importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title('Importancia de variables según LightGBM')
plt.xlabel('Importancia')
plt.ylabel('Variable')
plt.tight_layout()
plt.show()