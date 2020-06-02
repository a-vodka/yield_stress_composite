import numpy as np
import matplotlib.pyplot as plt
import const_proc

fiber_radius = 0.9  # fiber radius

e_fiber = 74.8e9  # Pa
nu_fiber = 0.2  # Possion ratio
g_fiber = 0.5 * e_fiber / (1 + nu_fiber)  # Shear modulus, Pa
yield_fiber = 170e6  # Yeild stress, Pa

e_matrix = 2.7e9
nu_matirx = 0.4
g_matrix = 1.5e9
yield_matrix = 80e6

width = 1
height = np.sqrt(3)

print("-----hexagonal_cell-----")
s_fiber = 2. / 4. * np.pi * fiber_radius ** 2
tot_area = width * height
s_matrix = tot_area - s_fiber
psi = s_fiber / tot_area
print("psi = {0}".format(psi))

e_matrix_rnd = np.random.normal(1, 0.05, 1000) * e_matrix

E_mix1 = e_fiber * psi + e_matrix_rnd * (1 - psi)
E_mix2 = (psi / e_fiber + (1 - psi) / e_matrix_rnd) ** -1

const_proc.plot_hist(e_matrix_rnd / 1e9, "E_{matrix}", pdf=const_proc.pdf_fit(e_matrix_rnd)[0:3])
const_proc.plot_hist(E_mix1 / 1e9, "E_{mix1}", pdf=const_proc.pdf_fit(E_mix1)[0:3])
const_proc.plot_hist(E_mix2 / 1e9, "E_{mix2}", pdf=const_proc.pdf_fit(E_mix2)[0:3])

plt.show()
