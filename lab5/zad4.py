import numpy as np
from matplotlib import pyplot as plt
import numpy.polynomial as poly


def get_chebyshev(n: int) -> list[float]:
	get_theta = lambda x: ((2*x + 1)/(2*n + 2))*np.pi
	return np.cos([get_theta(j) for j in range(n+1)])

def transform_chebyshev(cheb, a, b):
	return [a + (b-a)*(x+1)/2 for x in cheb]

f = lambda x: np.sqrt(x)

x_inter = transform_chebyshev(get_chebyshev(2), 0, 2)
y_inter = f(x_inter)

coefs = poly.chebyshev.chebfit(x_inter, y_inter, 2)
approx = poly.Polynomial(coefs)
print(approx)

x_new = np.linspace(0, 2, 100)
plt.scatter(x_inter, y_inter)
plt.plot(x_new, f(x_new), label='Funkcja właściwa')
plt.plot(x_new, approx(x_new), label='Aproksymacja')
plt.title('Aproksymacja funkcji sqrt(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
