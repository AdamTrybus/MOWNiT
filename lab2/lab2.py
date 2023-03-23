import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Wczytanie zbioru treningowego
train_data = pd.read_csv('breast-cancer-train.dat', header=None, delimiter=',')
train_labels = pd.read_csv('breast-cancer.labels', header=None)

# Dodanie nazw kolumn
train_data.columns = train_labels.iloc[:, 0].tolist()

# Wczytanie zbioru walidacyjnego
validate_data = pd.read_csv('breast-cancer-validate.dat', header=None, delimiter=',')
validate_labels = pd.read_csv('breast-cancer.labels', header=None)

# Dodanie nazw kolumn
validate_data.columns = validate_labels.iloc[:, 0].tolist()

# podpis osi i tytuł histogramu
plt.hist(validate_data.iloc[:,2], bins=15)
plt.xlabel('Radius')
plt.ylabel('Count')
plt.title('Histogram of Radius')

# podpis osi i tytuł wykresu
plt.figure()
plt.plot(validate_data.iloc[:,2], validate_data.iloc[:,4], 'k.')
plt.xlabel('Radius')
plt.ylabel('Perimeter')
plt.title('Scatter Plot of Radius vs Perimeter')

# wyświetlenie wykresów
plt.show()

#tworzenie reprezentacji liniowej
X_train_linear = train_data.iloc[:, 2:].values
X_validate_linear = validate_data.iloc[:, 2:].values

#tworzenie reprezentacji kwadratowej
X_train_quadratic = train_data.iloc[:, [2, 4, 5, 10]].values
X_validate_quadratic = validate_data.iloc[:, [2, 4, 5, 10]].values

quad_train = []
quad_val = []

for row in X_train_quadratic:
    quad_train.append([i for i in row] + [i*i for i in row] + [row[0]*row[1],row[0]*row[2],row[0]*row[3],row[1]*row[2],row[1]*row[3],row[2]*row[3]])

for row in X_validate_quadratic:
    quad_train.append([i for i in row] + [i*i for i in row] + [row[0]*row[1],row[0]*row[2],row[0]*row[3],row[1]*row[2],row[1]*row[3],row[2]*row[3]])


#Aby znaleźć wagi dla liniowej reprezentacji najmniejszych kwadratów, można skorzystać z równania normalnego, które ma postać:

b_vec_train = np.where(train_data.iloc[:, 1] == "M", 1, -1)

b_vec_val = np.where(validate_data.iloc[:, 1] == "M", 1, -1)

# obliczenie wag
A = X_train_linear
b = b_vec_train
w_linear = np.linalg.inv(A.T @ A) @ A.T @ b


# obliczenie wag
A2 = X_train_quadratic
b = b_vec_train
w_quad = np.linalg.inv(A2.T @ A2) @ A2.T @ b

#wspolczynniki cond powinny byc bliskie 1
cond = np.linalg.cond(A)
cond2 = np.linalg.cond(A2)

print(cond)
print(cond2)


#ostatni podpunkt

y_pred_linear = np.sign(X_validate_linear @ w_linear)
y_pred_quadratic = np.sign(X_validate_quadratic @ w_quad)
y_true = np.where(validate_data.iloc[:, 1] == "M", 1, -1)

fp_linear = np.sum((y_pred_linear == 1) & (y_true == -1))
fn_linear = np.sum((y_pred_linear == -1) & (y_true == 1))

# dla reprezentacji kwadratowej
fp_quadratic = np.sum((y_pred_quadratic == 1) & (y_true == -1))
fn_quadratic = np.sum((y_pred_quadratic == -1) & (y_true == 1))

print("False positives and false negatives for linear representation:")
print(f"False positives: {fp_linear}")
print(f"False negatives: {fn_linear}")
print("")

print("False positives and false negatives for quadratic representation:")
print(f"False positives: {fp_quadratic}")
print(f"False negatives: {fn_quadratic}")
