import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from matplotlib.patches import Ellipse


def func(x, a, b, c, d):
    return np.sqrt(1. + ((x - a) / b) ** 2) * c + d


def fitEllipse(cont, method):
    x = cont[:, 0]
    y = cont[:, 1]

    x = x[:, None]
    y = y[:, None]

    D = np.hstack([x * x, x * y, y * y, x, y, np.ones(x.shape)])
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))

    if method == 1:
        n = np.argmax(np.abs(E))
    else:
        n = np.argmax(E)
    a = V[:, n]

    # -------------------Fit ellipse-------------------
    b, c, d, f, g, a = a[1] / 2., a[2], a[3] / 2., a[4] / 2., a[5], a[0]
    num = b * b - a * c
    cx = (c * d - b * f) / num
    cy = (a * f - b * d) / num

    angle = 0.5 * np.arctan(2 * b / (a - c)) * 180 / np.pi
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    a = np.sqrt(abs(up / down1))
    b = np.sqrt(abs(up / down2))

    # ---------------------Get path---------------------
    ell = Ellipse((cx, cy), a * 2., b * 2., angle)
    ell_coord = ell.get_verts()

    params = [cx, cy, a, b, angle]

    return params, ell_coord


def plotConts(contour_list):
    '''Plot a list of contours'''
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    for ii, cii in enumerate(contour_list):
        x = cii[:, 0]
        y = cii[:, 1]
        ax2.plot(x, y, '-')
    plt.show(block=False)


outfile = "./yield_surface.csv"

res = np.loadtxt(outfile, delimiter=';', usecols=(0, 1))
print(res)
plt.scatter(res[:, 0], res[:, 1])
plt.show()

xdata = res[:, 0] / 1e6
ydata = res[:, 1] / 1e6

params1, ell1 = fitEllipse(res / 1e6, 1)
params2, ell2 = fitEllipse(res / 1e6, 2)

print(params1, ell1)

plt.scatter(res[:, 0]/1e6, res[:, 1]/1e6)
plt.plot(ell1[:, 0], ell1[:, 1])
plt.plot(ell2[:, 0], ell2[:, 1])

plt.show()