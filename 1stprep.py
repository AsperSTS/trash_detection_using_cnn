import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Definir el modelo
def create_multichannel_cnn(input_shape=(256, 256, 4), num_classes=6):

    """
    Crea un modelo de CNN multicanal.
    
    Args:
    input_shape: tuple, la forma de la entrada. Debe ser una tupla de 3 elementos, (alto, ancho, canales).
    num_classes: int, el número de clases a clasificar.
    
    Retorna:
    Un modelo de CNN multicanal.
    """
    

    model = Sequential()
    
    # Primera capa convolucional
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Segunda capa convolucional
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Tercera capa convolucional
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Cuarta capa convolucional
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Capa de aplanado
    model.add(Flatten())
    
    # Primera capa densa
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    # Segunda capa densa
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    # Capa de salida
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

# Crear el modelo
model = create_multichannel_cnn()

# Compilar el modelo
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()
