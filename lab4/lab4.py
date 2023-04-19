import numpy as np
from math import exp, cos, pi
import matplotlib.pyplot as plt
from scipy import interpolate

def f1(x):
    return 1 / (1 + 25*x**2)

def f2(x):
    return exp(cos(x))


def lagrange_interpolation(x, y, x_interp):
    n = len(x)
    m = len(x_interp)
    y_interp = np.zeros(m)

    for i in range(m):
        # Wyliczenie wartości wielomianu interpolacyjnego w punkcie x_interp[i]
        sum = 0
        for j in range(n):
            # Wyliczenie i-tego wielomianu Lagrange'a
            L = 1
            for k in range(n):
                if k != j:
                    L *= (x_interp[i] - x[k]) / (x[j] - x[k])
            sum += y[j] * L
        y_interp[i] = sum

    return y_interp


def transform(nodes,a,b):
    for i,node in enumerate(nodes):
        nodes[i] = a + (b-a)*(node+1)/2

def chebyshev_nodes(n):
    nodes = [0]*(n+1)
    for i in range(n+1):
        nodes[i] = cos((2*i+1)*pi/(2*n+2))
    return nodes

def calculate_error_norm(true_values, interpolated_values):
    errors = true_values - interpolated_values
    error_norm = np.linalg.norm(errors)
    return error_norm

n = 12
x = np.linspace(-1, 1, n)
nodes = chebyshev_nodes(n)

y = f1(x)
y_czeb = [f1(i) for i in nodes]


x_interp = np.linspace(-1, 1, n*10)
x_czeb_interp = np.linspace(-1, 1, n*10)

y_interp = lagrange_interpolation(x, y, x_interp)
y_czeb_interp = lagrange_interpolation(nodes, y_czeb, x_czeb_interp)

spl = interpolate.CubicSpline(x, y, bc_type='natural')
y_interp3 = spl(x_interp)

x_interp2 = np.linspace(0, 2*pi, n*10)
x2 = np.linspace(0, 2*pi, n)
y2 = [f2(i) for i in x2] 


plt.plot(x_interp, f1(x_interp), label='f1(x)')
plt.plot(x_interp, y_interp, label='Interpolacja')
plt.plot(x_interp,y_interp3,label='Interpolacja sklejana')
plt.plot(x_czeb_interp,y_czeb_interp,label='Interpolacja Czebyszewa')
plt.legend()
plt.xlabel('Wartości na osi x')
plt.ylabel('Wartości funkcji')
plt.show()

errors1 = []
errors2 = []
x_interp = np.random.uniform(-1, 1, 500)
x_interp2 = np.random.uniform(0, 2*pi, 500)
y_true = f1(x_interp)
y2_true = [f2(i) for i in x_interp2]
n_range = range(4,21)
for n in n_range:
    x = np.linspace(-1, 1, n)
    x_f2 = np.linspace(0, 2*pi, n)
    x_czeb = chebyshev_nodes(n)
    x_czeb_f2 = chebyshev_nodes(n)
    transform(x_czeb_f2,0,2*pi)

    y = f1(x)
    y_f2 = [f2(i) for i in x_f2]
    y_czeb = [f1(i) for i in x_czeb]
    y_czeb_f2 = [f2(i) for i in x_czeb_f2]

   


    y_interp1_1 = lagrange_interpolation(x, y, x_interp)
    y_interp2_1 = lagrange_interpolation(x_f2, y_f2, x_interp2)
    y_czeb_interp1_3 = lagrange_interpolation(x_czeb, y_czeb, x_interp)
    y_czeb_interp2_3 = lagrange_interpolation(x_czeb_f2, y_czeb_f2, x_interp2)

    spl = interpolate.CubicSpline(x, y, bc_type='natural')
    y_interp1_2 = spl(x_interp)

    spl2 = interpolate.CubicSpline(x_f2, y_f2, bc_type='natural')
    y_interp2_2 = spl2(x_interp2)

    errors1.append([calculate_error_norm(y_true, y_interp1_1), calculate_error_norm(y_true, y_interp1_2), calculate_error_norm(y_true, y_czeb_interp1_3)])
    errors2.append([calculate_error_norm(y2_true, y_interp2_1), calculate_error_norm(y2_true, y_interp2_2), calculate_error_norm(y2_true, y_czeb_interp2_3)])


errors1 = np.array(errors1).T
errors2 = np.array(errors2).T
fig, ax = plt.subplots()
width = 0.25
x_pos= np.arange(len(n_range))
ax.bar(x_pos, errors1[0], width=width, align='center', label="Lagrange")
ax.bar(x_pos +width , errors1[1], width=width, align='center', label="Sklejane")
ax.bar(x_pos + 2*width, errors1[2], width=width, align='center', label="Czebyszewa")
ax.set_xlabel('N-4')
ax.set_ylabel('Error')
ax.set_title('Interpolation error for f1(x)')
ax.set_xticks(x_pos+0.5)
ax.set_xticklabels(x_pos)
ax.legend()
plt.yscale('log')
plt.show()

fig, ax = plt.subplots()
width = 0.25
ax.bar(x_pos, errors2[0], width=width, align='center', label="Lagrange")
ax.bar(x_pos+width, errors2[1], width=width, align='center', label="Sklejane")
ax.bar(x_pos+ 2*width, errors2[2], width=width, align='center', label="Czebyszewa")
ax.set_xlabel('N-4')
ax.set_ylabel('Error')
ax.set_title('Interpolation error for f2(x)')
ax.set_xticks(x_pos+0.5)
ax.set_xticklabels(x_pos)
ax.legend()
plt.yscale('log')
plt.show()
