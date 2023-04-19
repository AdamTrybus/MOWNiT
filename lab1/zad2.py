import matplotlib.pyplot as plt
import numpy as np
def generuj_ciag(n):
    x = [1/3, 1/12] # wyrazy początkowe
    for i in range(2, n):
        xk = 2.25 * x[i-1] - 0.5 * x[i-2]
        x.append(xk)
    return x



n = 256
x = generuj_ciag(n)
blad = [0] * n
x_dokladne = np.array([4**(1-k)/3 for k in range(1,n+1)])
for i in range(n):
    blad[i] = abs((x[i] - x_dokladne[i]) / x_dokladne[i])

plt.semilogy(range(n), x)
plt.title("Wykres ciągu x[k]")
plt.xlabel("k")
plt.ylabel("xk")
plt.show()

plt.semilogy(range(n), blad)
plt.title("Wykres wartości bezwzględnej błędu względnego")
plt.xlabel("k")
plt.ylabel("| epsilon[k] |")
plt.show()

#catastrophic cancellation