from scipy.integrate import trapz, simpson
import numpy as np
import matplotlib.pyplot as plt

def midpoint_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a + h/2, b - h/2, n)
    return h * np.sum(f(x))

f = lambda x: 4/(1+(x**2))


# 1a)
max_m = 25
errors_mid = np.zeros(max_m)
errors_trap = np.zeros(max_m)
errors_simp = np.zeros(max_m)
ms = range(1,max_m+1)

for m in ms:
    h = 1/(2**m)
    x_ranges = np.linspace(0, 1, 2**m+1)
    midpoints = np.linspace(0+h, 1-h, 2**m)
    
    y_values = f(x_ranges)
    
    integral_mid = midpoint_rule(f,0,1,2**m)
    integral_trap = trapz(y_values, x_ranges)
    integral_simp = simpson(y_values, x_ranges)
    
    errors_mid[m-1] = abs((np.pi - integral_mid)/np.pi)
    errors_trap[m-1] = abs((np.pi - integral_trap)/np.pi)
    errors_simp[m-1] = abs((np.pi - integral_simp)/np.pi)


plt.figure(figsize=(8, 6))
plt.plot(ms, errors_mid, color="blue", label="błąd względny prost")
plt.plot(ms, errors_trap, color="red", label="błąd względny trap")
plt.plot(ms, errors_simp, color="green", label="błąd względny simp")
plt.title("Wykres błędów względnych dla odpowiednich kwadratur")
plt.xlabel('Liczba ewaluacji funkcji')
plt.ylabel('Błąd względny')
plt.yscale('log')
plt.legend(loc='best')
plt.show()

# 1b)
h_min_lab1 = np.finfo(float).eps ** 0.5
least_h = [m for m in range(max_m-1) if errors_simp[m] <= errors_simp[m+1]]

print("b) Porównanie wartości h_min")
print("h_min z lab1: ", h_min_lab1)
print("Wartości h_min, których zmniejszenie nie spowodowało zwiekszenie dokładności:")
print(*least_h, sep='\n')

# 1c)
h_1 = 2**5 +1 #wartości odczytane z wykresu
h_2 = 2**6 +1
ranges_1 = np.linspace(0, 1, h_1)
ranges_2 = np.linspace(0, 1, h_2)
y_1 = f(ranges_1)
y_2 = f(ranges_2)

mid_order = [
    midpoint_rule(f, 0, 1, h_1),
    midpoint_rule(f, 0, 1, h_2)
]
trap_order = [
    trapz(y_1, ranges_1),
    trapz(y_2, ranges_2)
]
simp_order = [
    simpson(y_1, ranges_1),
    simpson(y_2, ranges_2)
]
empirical_order_mid = (np.log(abs((mid_order[1] - mid_order[0])/(mid_order[0] - np.pi))))/(np.log(h_2/h_1))
empirical_order_trap = (np.log(abs((trap_order[1] - trap_order[0])/(trap_order[0] - np.pi))))/(np.log(h_2/h_1))
empirical_order_simp = (np.log(abs((simp_order[1] - simp_order[0])/(simp_order[0] - np.pi))))/(np.log(h_2/h_1))
print("Empiryczny rząd zbieżności midpoint to ", empirical_order_mid)
print("Empiryczny rząd zbieżności trap to ", empirical_order_trap)
print("Empiryczny rząd zbieżności simp to ", empirical_order_simp)

# 2
n_values = np.arange(2, 26, 1)
exact_value = np.pi
relative_errors = []

for n in n_values:
    x, w = np.polynomial.legendre.leggauss(n)  # węzły i wagi Legendre'a
    approx_value = np.sum(w * f(x)) / 2  # metoda Gaussa-Legendre'a
    rel_error = abs((approx_value - exact_value) / exact_value)
    relative_errors.append(rel_error)

plt.plot(n_values, relative_errors, color="black", label="Gauss-Legendre")
plt.plot(ms, errors_mid, color="blue", label="błąd względny prost")
plt.plot(ms, errors_trap, color="red", label="błąd względny trap")
plt.plot(ms, errors_simp, color="green", label="błąd względny simp")
plt.legend(loc='best')
plt.xlabel('Liczba ewaluacji funkcji')
plt.ylabel('Błąd względny')
plt.yscale('log')
plt.show()