import numpy as np
import matplotlib.pyplot as plt


# Dane z tabeli populacji stanów zjednoczonych
years = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980])
population = np.array([76212168, 92228496, 106021537, 123202624, 132164569, 151325798,179323175 ,203302031,226542199])

#Zad1 aproksymacja średniokwadratowa punktowa
# Przedział interpolacji
xmin, xmax = years[0], years[-1]

# Wartości x dla wykresu (odstępy równomierne)
x_plot = np.linspace(xmin, xmax, 1000)

# Wielomiany interpolacyjne
for m in range(7):
    coeffs = np.polyfit(years, population, m)
    y_plot = np.polyval(coeffs, x_plot)
    plt.plot(x_plot, y_plot, label=f'm = {m}')

# Dane oryginalne
plt.scatter(years, population, label='Dane oryginalne', color='black')

# Konfiguracja wykresu
plt.title('Aproksymacja średnio kwadratowa punktowa populacji Stanów Zjednoczonych')
plt.xlabel('Rok')
plt.ylabel('Populacja')
plt.legend(loc='upper left')

# Wyświetlenie wykresu
plt.show()

# a)
# Prawdziwa wartość populacji w 1990 roku
true_value = 248709873

# Ekstrapolacja dla każdego stopnia wielomianu
tab_coeffs = []
for m in range(7):
    coeffs = np.polyfit(years, population, m)
    tab_coeffs.append(coeffs)
    predicted_value = np.polyval(coeffs, 1990)
    error = (predicted_value - true_value) / true_value * 100
    print(f'm = {m}, Przewidywana wartość: {predicted_value}, Błąd względny: {error:.2f}%')

# Możemy zauważyć, że błąd względny ekstrapolacji maleje wraz ze wzrostem stopnia wielomianu,
# a najlepszy wynik (najmniejszy błąd względny) uzyskujemy dla m=4, czyli wielomianu czwartego stopnia.

#b)

def AIC(population, coeffs, k):
    n = len(population)
    residual_sum_sq = 0
    for i in range(len(population)):
        residual_sum_sq += np.square(population[i] - np.polyval(coeffs, years[i]))
    AIC = 2*k + n*np.log(residual_sum_sq/n)
    return AIC

def AICc(AIC, k, n):
    return AIC + (2*k*(k+1))/(n-k-1)

for m in range(len(coeffs)):
    print("m= ",m,", wartość kryterium: ",AICc(AIC(population,tab_coeffs[m],m+1),m+1,len(population)))

# Warto jednak zauważyć, że ujemne wartości AIC niekoniecznie oznaczają błąd w implementacji czy wadliwość metody.
# W przypadku gdy porównywane modele są bardzo dobre, wartości AIC dla wszystkich modeli mogą być ujemne, a różnica między nimi będzie jednakowa.
# Z tego powodu AIC powinno być stosowane raczej do porównywania różnych modeli, niż do oceny ich jakości w bezwzględny sposób.