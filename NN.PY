import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Datos de entrenamiento
data = {
    'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
    'feature2': [0.5, 0.4, 0.3, 0.2, 0.1],
    'label': [0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['feature1', 'feature2']].values 
y = df['label'].values 

# Definición del modelo
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))  
model.add(Dense(1, activation='sigmoid'))  

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(X, y, epochs=100, batch_size=1, verbose=1)

# Nuevos datos para predicción (varios ejemplos)
test_data = np.array([
    [0.2, 0.4],
    [0.3, 0.3],
    [0.5, 0.1],
    [0.1, 0.5]
])

# Predicción
predictions = model.predict(test_data)
predicted_labels = (predictions > 0.5).astype(int)

# Mostrar resultados
for i, label in enumerate(predicted_labels):
    print(f"Muestra {i+1} - Predicción: {label[0]} (Probabilidad: {predictions[i][0]:.4f})")


