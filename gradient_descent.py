
# En gradient descent, lo que estamos haciendo es dibujar una linea aleatoria que atraviesa
# nuestro set de datos (esta linea se grafica considerando todas nuestras variables independientes). 
# Lo que queremos es obtener la linea que mejor describa nuestros datos. 
# Cuando encontramos esta linea, tambien encontramos un coeficiente para cada una de nuestras variables
# (que tanto y como afecta cada variable a dicha linea), y un bias para la ecuacion (que tan arriba esta 
# la linea cuando los coeficientes son 0).

# Como mencione la linea comienza siendo aleatoria, y con gradient descent lo que hacemos es medir la distancia
# que hay entre cada punto (cada dato) y nuestra linea predictiva. Nuestro trabajo es hacer que esta distancia sea 
# la menor posible.

# MSE = Mean squared error
# delta i = la diferencia entre la linea y el valor real. Tomamos todas las deltas las elevamos al cuadrado para usar
# solo valores positivos y que no se cancelen (con valores negativos) y las dividimos en el numero de puntos para saber cuanto
# a esto se le conoce como el mean squared error. Mientras menor sea este valor, mas cerca estara nuestra prediccion de la realidad
# (al menos usualmente).

# Lo que queremos es ver como cada una de nuestros coeficientes  m y nuestro bias b afectan este mean squared error. Para ello 
# derivamos MSE con respecto a cada m y al b.

# Esta derivada nos dice la direccion a la que cada bias y coeficiente esta moviendo nuestro MSE, la idea es irnos moviendo hacia el MSE mas pequenio posible. 
# Esto se consigue utilizando un step (que es lo mismo que el learning rate), que nos dice que tanto movernos hacia la direccion que hace que el MSE disminuya.

import matplotlib.pyplot as plt
import numpy as np


def gradient_descent(X, y, learning_rate=0.001, iterations=25000):
    # Convertir a numpy arrays
    X = np.array(X)
    y = np.array(y)

    n_samples, n_features = X.shape  # filas, columnas

    # Inicializar coeficientes (w) y bias (b)
    w = np.zeros(n_features)
    b = 0

    for i in range(iterations):
        # Prediccion del modelo
        y_pred = np.dot(X, w) + b   # X*w + b

        # Calcular costo (MSE)
        cost = (1/n_samples) * np.sum((y - y_pred) ** 2)

        # Derivadas parciales
        dw = -(2/n_samples) * np.dot(X.T, (y - y_pred))  # vector de derivadas
        db = -(2/n_samples) * np.sum(y - y_pred)

        # Actualizar pesos
        w -= learning_rate * dw
        b -= learning_rate * db

        # Mostrar progreso cada cierto número de iteraciones
        if i % 500 == 0:
            print(f"Iteración {i}: w={w}, b={b:.4f}, cost={cost:.4f}")

    return w, b

# Cargar "abalone.data" 
def load_data(filename="abalone.data"):
    data = []
    with open(filename, "r") as f:
        for line in f:
            #separar valores por comas
            values = line.strip().split(",")
            data.append(values)
    #convertir las lineas de datos en un array
    return np.array(data)

# Hot encoding para "Sex" (Male, Female, Infant)
def encode_sex(data):
    encoded = []
    for row in data:
        sex = row[0] #sex es el primer valor de la fila
        if sex == "M":
            encoded.append([1, 0, 0])  # M
        elif sex == "F":
            encoded.append([0, 1, 0])  # F
        else:
            encoded.append([0, 0, 1])  # I
    
    return np.array(encoded)

# Separar variables independientes (X) y dependiente (y)
def preprocess_data(raw_data):
    # Separar columna sex (cualitativa) y el resto
    sex_encoded = encode_sex(raw_data)

    # Convertir todo a float para calculos
    numeric_data = raw_data[:, 1:].astype(float)  

    # Nuestra x son las independientes sin el numero de rings (y)
    #hstack permite juntar las 3 columnas del sex_encoded con el resto 
    X = np.hstack((sex_encoded, numeric_data[:, :-1]))  

    # Variable dependiente = rings
    y = numeric_data[:, -1]  
    return X, y

# Separar en entrenamiento (80%) y test (20%)
def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_size = int(n_samples * test_size)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test

# Separar en entrenamiento (60%) y test (20%) y validacion 20%)
def train_val_test_split(X, y, val_size=0.2, test_size=0.2, seed=42):
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_split = int(n_samples * test_size)
    val_split = int(n_samples * (test_size + val_size))

    test_idx = indices[:test_split]
    val_idx = indices[test_split:val_split]
    train_idx = indices[val_split:]

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test

# Diagnostico del modelo basado en R2 de train y val
def diagnostico_modelo(y_train, y_train_pred, y_val, y_val_pred):
    r2_train = r_squared(y_train, y_train_pred)
    r2_val = r_squared(y_val, y_val_pred)

    # Bias
    if r2_train < 0.3:
        bias = "Alto (modelo no aprende bien)"
    elif r2_train < 0.7:
        bias = "Medio (modelo mejora pero sigue limitado)"
    else:
        bias = "Bajo (modelo aprende bien)"

    # Varianza
    diff = abs(r2_train - r2_val)
    
    if diff > 0.2:
        varianza = "Alta (cambia mucho entre train y val)"
    elif diff > 0.1:
        varianza = "Media (algo de diferencia)"
    else:
        varianza = "Baja (modelo consistente)"

    # Ajuste
    if r2_train < 0.3:
        ajuste = "Underfit (el modelo no captura los patrones)"
    elif r2_train >= 0.7 and diff > 0.2:
        ajuste = "Overfit (memoriza train pero falla en val)"
    else:
        ajuste = "Fit (buen balance entre bias y varianza)"

    return {
        "R2_train": r2_train,
        "R2_val": r2_val,
        "Bias": bias,
        "Varianza": varianza,
        "Ajuste": ajuste
    }


def run_Abalone():
    raw_data = load_data("abalone.data")
    X, y = preprocess_data(raw_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    w, b = gradient_descent(X_train, y_train, learning_rate=0.001, iterations=20000)
    print("Pesos finales:", w)
    print("Bias final:", b)

    return w, b, X_test, y_test

def run_Abalone_Validation():
    raw_data = load_data("abalone.data")
    X, y = preprocess_data(raw_data)
    
    # usar el split con validación
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # entrenar el modelo
    w, b = gradient_descent(X_train, y_train, learning_rate=0.001, iterations=20000)

    # predicciones
    y_train_pred = predict(X_train, w, b)
    y_val_pred = predict(X_val, w, b)
    y_test_pred = predict(X_test, w, b)

    # diagnóstico
    diag = diagnostico_modelo(y_train, y_train_pred, y_val, y_val_pred)
    print("\n--- Diagnóstico del modelo ---")
    for k, v in diag.items():
        print(f"{k}: {v}")

    # desempeño en test final
    r2_test = r_squared(y_test, y_test_pred)
    print(f"\nR2 en test set (generalización): {r2_test:.4f}")

    return w, b, (y_train, y_train_pred), (y_val, y_val_pred), (y_test, y_test_pred)

# Función de estandarización
def standardize(X_train, X_val=None, X_test=None):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_scaled = (X_train - mean) / std
    
    results = [X_train_scaled]
    if X_val is not None:
        X_val_scaled = (X_val - mean) / std
        results.append(X_val_scaled)
    if X_test is not None:
        X_test_scaled = (X_test - mean) / std
        results.append(X_test_scaled)
    return tuple(results)

def run_Abalone_Validation_Standardized():
    raw_data = load_data("abalone.data")
    X, y = preprocess_data(raw_data)
    
    # Split train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    
    # Estandarizar los datos
    X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    # Entrenar modelo sin regularización
    w, b = gradient_descent(
        X_train, y_train,
        learning_rate=0.001,
        iterations=20000
    )

    # Predicciones
    y_train_pred = predict(X_train, w, b)
    y_val_pred = predict(X_val, w, b)
    y_test_pred = predict(X_test, w, b)

    # Diagnóstico
    diag = diagnostico_modelo(y_train, y_train_pred, y_val, y_val_pred)
    print("\n--- Diagnóstico del modelo ESTANDARIZADO ---")
    for k, v in diag.items():
        print(f"{k}: {v}")

    # R² en test
    r2_test = r_squared(y_test, y_test_pred)
    print(f"\nR2 en test set (generalización): {r2_test:.4f}")

    return w, b, (y_train, y_train_pred), (y_val, y_val_pred), (y_test, y_test_pred)

# Graficar resultados
def plot_results(y_true, y_pred, title):
    plt.figure(figsize=(6,5))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)  
    r2 = r_squared(y_true, y_pred)
    plt.xlabel("Valores reales")
    plt.ylabel("Valores predichos")
    plt.title(f"{title} (R2={r2:.4f})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()



#Probar que funciona
#modelo de datos random grande el cual modela 2x+3z+5n+4:
def run_test_code():
    np.random.seed(42)
    X = np.random.randint(0, 10, (1000, 4))
    y = 2*X[:,0] + 3*X[:,1] + 5*X[:,2] + 7*X[:,3] + 4
    #El resultado debe ser 2, 3, 5, 7, 4 y el cost de casi 0, el Rsquared debe ser 1 o cercano
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    w, b = gradient_descent(X_train, y_train, learning_rate=0.001, iterations=20000)
    
    print("Pesos finales:", w)
    print("Bias final:", b)
    
    return w, b, X_test, y_test




# Predecir con los coeficientes obtenidos
def predict(X, w, b):
    return np.dot(X, w) + b

# Calcular R squared
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) 
    return 1 - (ss_res / ss_tot)

# Hacer predicciones sobre el set de test

def main():
    print("Escribe 1, 2, 3 o 4")
    eleccion = input("1 - Correr con el dataset de abalone \n2 - Correr con un set de prueba\n"
                     "3 - Correr abalone con set de validacion\n4 - Correr abalone con validación + standarización\nEleccion: ")

    if eleccion == "1":
        w, b, X_test, y_test = run_Abalone()
        y_pred_test = predict(X_test, w, b)
        plot_results(y_test, y_pred_test, "Test Set")

    elif eleccion == "2":
        w, b, X_test, y_test = run_test_code()
        y_pred_test = predict(X_test, w, b)
        plot_results(y_test, y_pred_test, "Test Set")

    elif eleccion == "3":
        w, b, (y_train, y_train_pred), (y_val, y_val_pred), (y_test, y_test_pred) = run_Abalone_Validation()
        plot_results(y_train, y_train_pred, "Train Set")
        plot_results(y_val, y_val_pred, "Validation Set")
        plot_results(y_test, y_test_pred, "Test Set")

    elif eleccion == "4":
        w, b, (y_train, y_train_pred), (y_val, y_val_pred), (y_test, y_test_pred) = run_Abalone_Validation_Standardized()
        plot_results(y_train, y_train_pred, "Train Set (Estandarizado)")
        plot_results(y_val, y_val_pred, "Validation Set (Estandarizado)")
        plot_results(y_test, y_test_pred, "Test Set (Estandarizado)")

    else:
        print("Eleccion no valida")
        return


main()
