from notebook_grader import BicycleSolution, grade_bicycle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Bicycle():
    def __init__(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0

        self.L = 2
        self.lr = 1.2
        self.w_max = 1.22

        self.sample_time = 0.01

    def reset(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0

##--------------------------------------------------------------------------------------------------------------------##
class Bicycle(Bicycle):
    def step(self, v, w):
        w = max(-self.w_max, min(w, self.w_max))
        xc_dot = v * np.cos(self.beta + self.theta)
        yc_dot = v * np.sin(self.beta + self.theta)
        theta_dot = v * np.cos(self.beta) * np.tan(self.delta) / self.L
        delta_dot = w
        self.xc += xc_dot * self.sample_time
        self.yc += yc_dot * self.sample_time
        self.delta += delta_dot * self.sample_time
        self.theta += theta_dot * self.sample_time
        self.beta = np.arctan(self.lr * np.tan(self.delta) / self.L)
##--------------------------------------------------------------------------------------------------------------------##

sample_time = 0.01
time_end = 20
model = Bicycle()
solution_model = BicycleSolution()

# set delta directly
model.delta = np.arctan(2 / 10)
solution_model.delta = np.arctan(2 / 10)

t_data = np.arange(0, time_end, sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
x_solution = np.zeros_like(t_data)
y_solution = np.zeros_like(t_data)

for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    model.step(np.pi, 0)

    x_solution[i] = solution_model.xc
    y_solution[i] = solution_model.yc
    solution_model.step(np.pi, 0)

    model.beta = 0
    # solution_model.beta=0

plt.axis('equal')
plt.plot(x_data, y_data, label='Learner Model')
plt.plot(x_solution, y_solution, label='Solution Model')
plt.legend()
plt.show()


##--------------------------------------------------------------------------------------------------------------------##
#sample for beginner(square, spiral, wave)

sample_time = 0.01
time_end = 60
model.reset()
solution_model.reset()

t_data = np.arange(0, time_end, sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
x_solution = np.zeros_like(t_data)
y_solution = np.zeros_like(t_data)

# maintain velocity at 4 m/s
v_data = np.zeros_like(t_data)
v_data[:] = 4

w_data = np.zeros_like(t_data)

# ==================================
#  Square Path: set w at corners only
# ==================================
# w_data[670:670+100] = 0.753
# w_data[670+100:670+100*2] = -0.753
# w_data[2210:2210+100] = 0.753
# w_data[2210+100:2210+100*2] = -0.753
# w_data[3670:3670+100] = 0.753
# w_data[3670+100:3670+100*2] = -0.753
# w_data[5220:5220+100] = 0.753
# w_data[5220+100:5220+100*2] = -0.753

# ==================================
#  Spiral Path: high positive w, then small negative w
# ==================================
# w_data[:] = -1/100
# w_data[0:100] = 1

# ==================================
#  Wave Path: square wave w input
# ==================================
w_data[:] = 0
w_data[0:100] = 1
w_data[100:300] = -1
w_data[300:500] = 1
w_data[500:5700] = np.tile(w_data[100:500], 13)
w_data[5700:] = -1

# ==================================
#  Step through bicycle model
# ==================================
for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    model.step(v_data[i], w_data[i])

    x_solution[i] = solution_model.xc
    y_solution[i] = solution_model.yc
    solution_model.step(v_data[i], w_data[i])

plt.axis('equal')
plt.plot(x_data, y_data, label='Learner Model')
plt.plot(x_solution, y_solution, label='Solution Model')
plt.legend()
plt.show()

##--------------------------------------------------------------------------------------------------------------------##
#반지름 8m, 또한 경로를 30s안에 끝내야 한다. 밑의 코드는 이를 위한 코드이다.

sample_time = 0.01
time_end = 30
model.reset()

t_data = np.arange(0,time_end,sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
v_data = np.zeros_like(t_data)
w_data = np.zeros_like(t_data)


radius = 8 #원의 반지름

v_data[:] = 2 * (2 * radius * np.pi) / time_end  #원이 두 개이므로 *2를 해준다.

delta = 0.95 * np.arctan(model.L/radius)  #delta 구하는 공식 대입

for i in range(t_data.shape[0]):  #모델에 대입하고 반복문을 통해 채워 넣기
    x_data[i] = model.xc
    y_data[i] = model.yc

    if i <= t_data.shape[0]/8:
        if model.delta < delta:
            model.step(v_data[i], model.w_max)
            w_data[i] = model.w_max
        else:
            model.step(v_data[i], 0)
            w_data[i] = 0

    elif i <= 5.05 * t_data.shape[0]/8:
        if model.delta > -delta:
            model.step(v_data[i], -model.w_max)
            w_data[i] = -model.w_max
        else:
            model.step(v_data[i], 0)
            w_data[i] = 0

    else:
        if model.delta < delta:
            model.step(v_data[i], model.w_max)
            w_data[i] = model.w_max
        else:
            model.step(v_data[i], 0)
            w_data[i] = 0

plt.axis('equal')
plt.plot(x_data, y_data)
plt.show()

##--------------------------------------------------------------------------------------------------------------------##
#밑의 코드는 검증 코드
grade_bicycle(t_data,v_data,w_data)