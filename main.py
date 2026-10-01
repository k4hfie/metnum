import numpy as np 
import matplotlib.pyplot as plt

windspeed = np.arange(3.0, 20.5, 0.5)

power = np.array([
    0.0, 0.6, 1.8, 3.9, 7.2,
    11.8, 17.6, 24.9, 34.2, 45.7,
    59.4, 75.8, 92.7, 110.9, 129.6,
    148.2, 168.4, 187.6, 202.5, 215.7,
    226.8, 235.1, 241.3, 245.6, 248.1,
    249.2, 250.0, 249.7, 250.0, 249.8,
    250.0, 249.9, 250.0, 249.8, 250.0 
])

def polynomial_regression(x, y):
    num_data = len(x)
    degree = 2

    sum_x  = np.sum(x)
    sum_x2 = np.sum(x**2)
    sum_x3 = np.sum(x**3)
    sum_x4 = np.sum(x**4)

    sum_y   = np.sum(y)
    sum_xy  = np.sum(x*y)
    sum_x2y = np.sum((x**2)*y)

    matrix_A = np.array([
       [num_data, sum_x, sum_x2],
       [sum_x, sum_x2, sum_x3],
       [sum_x2, sum_x3, sum_x4]     
    ])

    matrix_B = np.array([
        sum_y, sum_xy, sum_x2y
    ])

    a0, a1, a2 = np.linalg.solve(matrix_A, matrix_B)

    def model(x_input):
        return a0 + a1*x_input + a2*(x_input**2)

    return model

def plot(x, y_measured, y_prediction, title):
    plt.figure(figsize=(8,5))
    plt.scatter(x, y_measured, color='red', label='measured')
    plt.plot(x, y_prediction, color='blue', label='prediction')
    
    plt.xlim(0, 20.5)
    plt.ylim(0, 300)
 
    plt.title(title)
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Power Output (kW)')
    plt.legend()
    plt.show()

def main():
    base_model = polynomial_regression(windspeed, power)
    power_prediction = base_model(windspeed)
    plot(windspeed, power, power_prediction, 'Polynomial Regression Model')

main()