# Librerías
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from scipy.stats import zscore

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









