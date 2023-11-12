import numpy as np
import matplotlib.pylab as plt

def line(a, b, x):
    return a*x+b

x = np.linspace(0, 10)
a = 2; b = 3
y = line(a, b, x)

size = 333
x_sample = np.linspace(0, 10, size)
y_sample = a * x_sample + b + np.random.normal(loc=0, scale=2, size=size)

plt.scatter(x_sample, y_sample, marker='o', alpha=0.6, c='r')
plt.plot(x, y, c='k', lw=2.)
plt.xlim(0, 10); plt.ylim(0, 30)
plt.xlabel('x'); plt.ylabel('y')

size = 1000
x_sample = np.linspace(0, 10, size)
x_class = np.linspace(0, 10, size)
y_class0 = np.array([r if r <= line(a, b, x) + 3 else -1 for x,r in zip(x_sample, 30 * np.random.uniform(size=size))])
y_class1 = np.array([r if r >= line(a, b, x) - 3 else -1 for x,r in zip(x_sample, 30 * np.random.uniform(size=size))])

plt.scatter(x_class[:len(y_class0)], y_class0, 	marker='o',alpha=0.6, c='r')
plt.scatter(x_class[:len(y_class1)], y_class1, 	marker='o',alpha=0.6, c='b')
plt.plot(x, y, c='k', lw=2.)
plt.xlim(0, 10); plt.ylim(0, 30)
plt.xlabel('x'); plt.ylabel('y')

fig = plt.figure(1, figsize=(9, 6))

plt.scatter(x_class,y_class0, marker='o',alpha=0.6, c='r')
plt.scatter(x_class,y_class1, marker='o',alpha=0.6, c='b')
plt.plot(x,y, c='k',lw=50.,alpha=0.3) # fat transparent line over opaque regression line
plt.plot(x,y, c='k',lw=2.,alpha=0.9) # regression line
plt.xlim(0,10)
plt.ylim(0,30)
plt.xlabel('x')
plt.ylabel('y')

plt.show()