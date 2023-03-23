import math
import matplotlib.pyplot as plt
import numpy

def approx_derivative(f, x, h):
    return (f(x+h) - f(x)) / h

def approx_derivative2(f, x, h):
    return (f(x+h) - f(x-h)) / (h*2)

def f(x):
    return math.tan(x)

def sec2(x):
    return 1 + (math.tan(x)) **2

h_values = [10**(-k) for k in range(17)]
error_values = []
x = 1
eps = math.sqrt(numpy.finfo(float).eps)
print("eps - " + str(eps))
for h in h_values:
    approx_value = approx_derivative(f, x, h) # tutaj zmieniamy funkcje na approx_derivative2, w momencie gdy chcemy skorzystać z wzoru różnic centralnych
    error = abs(approx_value - sec2(x))
    error_values.append(error)

plt.figure(figsize=(8,6))
plt.loglog(h_values, error_values)
plt.xlabel('h')
plt.ylabel('Absolute error')
plt.title('Absolute error of the approximated derivative of tan(x)')
plt.show()