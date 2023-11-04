import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tkinter as tk
from tkinter import ttk

# Loading MNIST dataset using TensorFlow
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize pixel values to [0, 1]

# Visualizing the loaded MNIST dataset
plt.imshow(X_train[0], cmap='binary')
plt.show()

# Build and train a CNN model using TensorFlow and Keras
model = models.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    layers.Conv2D(32, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Visualize confusion matrix
y_pred = model.predict_classes(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Create a simple Tkinter GUI to predict using the trained model
def predict():
    try:
        index = int(entry_index.get())
        if 0 <= index < len(X_test):
            input_data = X_test[index][None]
            prediction = model.predict(input_data)
            result_label.config(text=f"Predicted Digit: {np.argmax(prediction[0])}")
        else:
            result_label.config(text="Invalid Index")
    except ValueError:
        result_label.config(text="Invalid Index")

root = tk.Tk()
root.title("MNIST Digit Prediction")
root.geometry("300x150")

frame = ttk.Frame(root)
frame.pack(padx=10, pady=10, fill='both', expand=True)

label_index = ttk.Label(frame, text="Test Image Index:")
label_index.grid(row=0, column=0)
entry_index = ttk.Entry(frame)
entry_index.grid(row=0, column=1)

predict_button = ttk.Button(frame, text="Predict Digit", command=predict)
predict_button.grid(row=1, columnspan=2)

result_label = ttk.Label(frame, text="")
result_label.grid(row=2, columnspan=2)

root.mainloop()
