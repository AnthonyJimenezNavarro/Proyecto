# Librerías
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance

# Cargar datos
data = 'data/ObesityDataSet_raw_and_data_sinthetic.xlsx'
df = pd.read_excel(data)
head = df.head()
head.to_excel('res/tables/head.xlsx', index=False)

# Variables
variables_cuantitativas = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
resumen_metricas = df[variables_cuantitativas].describe().loc[['min', '25%', '50%', '75%', 'max', 'mean']]
resumen_metricas.to_excel('res/tables/resumen.xlsx', index=True)

# Preparación para modelo
objetivo = 'NObeyesdad'
X = df.drop(columns=[objetivo])
y = df[objetivo]

# Codificar variables categóricas
cat_cols = X.select_dtypes(include='object').columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Codificar variable objetivo
le_objetivo = LabelEncoder()
y_encoded = le_objetivo.fit_transform(y)

# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Modelo XGBoost
xgb_clf = XGBClassifier(
    objective='multi:softmax',
    num_class=len(le_objetivo.classes_),
    learning_rate=0.05,
    n_estimators=100,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# Entrenamiento
xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Predicciones
y_pred_labels = xgb_clf.predict(X_test)

# Evaluación
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_labels, target_names=le_objetivo.classes_))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_labels))

# Importancia de variables
plt.figure(figsize=(10, 6))
plot_importance(xgb_clf, max_num_features=15, importance_type='weight')
plt.title('Importancia de variables según XGBoost')
plt.tight_layout()
plt.show()
