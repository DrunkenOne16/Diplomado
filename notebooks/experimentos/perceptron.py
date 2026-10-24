import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

class PerceptronVisual:
    def __init__(self, learning_rate=1.0, epochs=6):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(2)
        self.bias = 0.0

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        return self.activation(np.dot(inputs, self.weights) + self.bias)

    def dibujar(self, X, y, title, epoch, fig_ax):
        ax = fig_ax
        ax.clear()

        # Malla para el mapa de calor
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 300),
                             np.linspace(-0.5, 1.5, 300))
        Z = self.weights[0] * xx + self.weights[1] * yy + self.bias

        # Escala simétrica para que el 0 siempre sea el color neutro
        lim = max(abs(Z.min()), abs(Z.max()), 1e-6)
        cs = ax.contourf(xx, yy, Z, levels=np.linspace(-lim, lim, 25),
                         cmap="coolwarm", alpha=0.85)
        # Líneas de nivel finas
        ax.contour(xx, yy, Z, levels=np.linspace(-lim, lim, 13),
                   colors="white", linewidths=0.6, alpha=0.7)
        # Frontera de decisión (Z = 0)
        ax.contour(xx, yy, Z, levels=[0], colors="black",
                   linewidths=3, linestyles="--")

        # Puntos de datos
        for i, s in enumerate(X):
            if y[i] == 1:
                ax.scatter(s[0], s[1], c="gold", edgecolors="black",
                           s=250, marker="o", linewidths=2, zorder=5)
            else:
                ax.scatter(s[0], s[1], c="black", s=250, marker="X",
                           edgecolors="white", linewidths=1.5, zorder=5)
            ax.text(s[0] + 0.07, s[1] + 0.07, f"({s[0]},{s[1]})={y[i]}",
                    fontsize=10, fontweight="bold", zorder=6,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_title(f"{title}\nÉpoca {epoch} | W1={self.weights[0]:.2f}, "
                     f"W2={self.weights[1]:.2f}, b={self.bias:.2f}")
        ax.set_aspect("equal")
        return cs

    def train_and_plot(self, X, y, title):
        # Pesos iniciales pequeños
        self.weights = np.array([0.1, -0.1])
        self.bias = -0.1

        fig, ax = plt.subplots(figsize=(7, 6))

        # Estado inicial (época 0)
        cs = self.dibujar(X, y, title, 0, ax)
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label("w·x + b  (rojo → clase 1, azul → clase 0)")
        plt.pause(1.0)

        for epoch in range(1, self.epochs + 1):
            error_flag = False
            for xi, target in zip(X, y):
                pred = self.predict(xi)
                error = target - pred
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    error_flag = True

            self.dibujar(X, y, title, epoch, ax)
            plt.pause(0.8)

            if not error_flag:
                print(f"[{title}] ¡Convergencia alcanzada en la época {epoch}!")
                break
        else:
            print(f"[{title}] No convergió en {self.epochs} épocas.")

        plt.show()

# --- DATOS ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# 1. Compuerta AND
y_and = np.array([0, 0, 0, 1])
perceptron_and = PerceptronVisual(learning_rate=1.0, epochs=6)
print("Ejecutando visualización para COMPUERTA AND...")
perceptron_and.train_and_plot(X, y_and, "Compuerta AND (Linealmente Separable)")

# 2. Compuerta XOR
y_xor = np.array([0, 1, 1, 0])
perceptron_xor = PerceptronVisual(learning_rate=1.0, epochs=6)
print("\nEjecutando visualización para COMPUERTA XOR...")
perceptron_xor.train_and_plot(X, y_xor, "Compuerta XOR (No Linealmente Separable)")



