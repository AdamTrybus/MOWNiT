import numpy as np
import matplotlib.pyplot as plt

# Dane z tabeli populacji stanów zjednoczonych
years = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980])
population = np.array([76212168, 92228496, 106021537, 123202624, 132164569, 151325798,179323175 ,203302031,226542199])

# stopień wielomianu interpolacyjnego
n = len(years) - 1

# a) Tworzenie macierzy Vandermonde'a
V = np.vander(years, n+1, increasing=True)

# b)
cond = np.linalg.cond(V)
print("cond : ",cond)

# c) obliczanie współczynników wielomianu interpolacyjnego
coeffs = np.polyfit(years, population, deg=8)

# definiowanie wielomianu interpolacyjnego
p = np.poly1d(coeffs)

# przedział dla wykresu
x = np.arange(1900, 1990, 1)

# obliczanie wartości wielomianu interpolacyjnego na przedziale [1900,1990] w odstępach jednorocznych
values = []
for year in x:
    value = p[0]
    for i in range(1, 9):
        value = value * (year - years[-i]) + p[i]
    values.append(value)

# rysowanie wykresu
plt.plot(years, population, 'o', label='węzły interpolacji')
plt.plot(x, p(x), label='wielomian interpolacyjny')
plt.xlabel('Rok')
plt.ylabel('Populacja')
plt.legend()
plt.show()

# d) ekstrapolacja wielomianu do roku 1990
y1990 = coeffs[-1] * (1990 - years[-1])**len(coeffs)

true_value = 248709873
error = abs(true_value - y1990) / abs(true_value)
print("błąd względny : ",error)

# e) 
def lagrange_interpolation(x, y, xi):
    n = len(x)
    yi = 0.0
    for i in range(n):
        numerator, denominator = 1.0, 1.0
        for j in range(n):
            if j != i:
                numerator *= xi - x[j]
                denominator *= x[i] - x[j]
        yi += y[i] * numerator / denominator
    return yi

x = np.arange(1900, 1991).astype(np.float64)
y = np.zeros_like(x)

for i in range(len(x)):
    y[i] = lagrange_interpolation(years, population, x[i])
print("lagrange : ",y)
# f)
def newton_interp(x_values, y_values, x):
    n = len(x_values)
    coeffs = np.zeros(n)
    coeffs[0] = y_values[0]
    for i in range(1, n):
        for j in range(n-1, i-1, -1):
            y_values[j] = (y_values[j] - y_values[j-1]) / (x_values[j] - x_values[j-i])
        coeffs[i] = y_values[i]
    y = coeffs[-1]
    for i in range(n-2, -1, -1):
        y = coeffs[i] + (x - x_values[i]) * y
    return y

x = np.arange(1900, 1991).astype(np.float64)
y_interp = np.array([newton_interp(years, population, xi) for xi in x])
print("newton : ",y_interp)

# g)
rounded_population = np.round(population, decimals=-6)
rounded_coeffs = np.polyfit(years, rounded_population, deg=8)
print("zaokrąglone dane : ",rounded_coeffs)
