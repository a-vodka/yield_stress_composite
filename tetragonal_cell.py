import numpy as np
import matplotlib.pyplot as plt
from pyAPDLWrapper.ansyswrapper import ansyswrapper
from numpy import linalg as LA

fiber_radius = 0.8  # fiber radius
cell_width = 1.
cell_height = 1.

e_fiber = 74800e6  # Pa
nu_fiber = 0.2  # Possion ratio

g_fiber = 31000e6  # Shear modulus, Pa
yield_fiber = 170e6  # Yeild stress, Pa
e_matrix = 4200e6
nu_matirx = 0.4
g_matrix = 1500e6
yield_matrix = 80e6


def main(ex, ey, filename):
    psi = np.pi * fiber_radius ** 2 / cell_width / cell_height / 4.0
    # print("psi = {0}".format(psi))

    E_mix = e_fiber * psi + e_matrix * (1 - psi)
    E_mix2 = (psi / e_fiber + (1 - psi) / e_matrix) ** -1

    yield_mix = yield_fiber * psi + yield_matrix * (1 - psi)
    yield_mix2 = (psi / yield_fiber + (1 - psi) / yield_matrix) ** -1

    # print("Emix = {0:G}, Emix2 = {1:G},".format(E_mix, E_mix2))
    # print("yield_mix = {0:G}, yield_mix2 = {1:G},".format(yield_mix, yield_mix2))

    projdir = r'd:\ans_proj\compo_yield_stress'

    ans = ansyswrapper(projdir=projdir, jobname='myjob', anslic='aa_t_i', )
    ans.setFEByNum(183)

    e_matrix_rnd = np.random.normal(1, 0.2, 1)[0] * e_matrix
    e_fiber_rnd = np.random.normal(1, 0.2, 1)[0] * e_fiber
    print('e_matrix_rnd = {0}'.format(e_matrix_rnd))

    matrix_id = ans.createIsotropicMat(E=e_matrix_rnd, nu=nu_matirx)
    fiber_id = ans.createIsotropicMat(E=e_fiber_rnd, nu=nu_fiber)

    ans.rectangle(0, 0, cell_width, cell_height)
    ans.circle(0, 0, fiber_radius)
    ans.overlapAreas()
    ans.delOuterArea(0, 0, cell_width, cell_height)
    ans.setCirlceAreaMatProps(fiber_radius, matId=fiber_id)
    ans.mesh()

    # ans.applyTensX(0, 0, cell_width, cell_height)
    # ans.applyTensY(0, 0, cell_width, cell_height)
    # ans.applyTensXandY(0, 0, cell_width, cell_height)
    # ans.applyShearXY(0, 0, cell_width, cell_height, eps=ex)
    # ans.precessElasticConstants()

    ans.applyTensXandY(0, 0, cell_width, cell_height, epsx=ex, epsy=ey)
    ans.saveMaxStressForEachMaterial()
    ans.saveToFile(projdir + '\\test.apdl')
    retcode = ans.run()

    if (retcode > 0):
        print('retcode = {0}'.format(retcode))
        exit(retcode)

    s, e = ans.getAVGStressAndStrains()
    s_principals, _ = LA.eig(s)

    maxs = ans.getMaxStressForEachMaterial() / np.array([yield_matrix, yield_fiber])

    # print(maxs)
    yield_stress = s_principals / np.min(maxs)

    print(yield_stress)

    out_file = open(filename, mode='a')
    np.savetxt(out_file, np.reshape(yield_stress, (1, 3)), delimiter=';', newline='\n')
    out_file.close()

    pass


outfile = "./yield_surface.csv"
ex = np.linspace(-0.1, 0.1, 4, endpoint=True)
ey = np.linspace(-0.1, 0.1, 4, endpoint=True)

xv, yv = np.meshgrid(ex, ey, sparse=False, indexing='ij')
for i in range(10):
    for j in range(10):
        #print(np.random.uniform(-0.1, 0.1, 1)[0], np.random.uniform(-0.1, 0.1, 1)[0])
        for k in range(5):
            main(np.random.uniform(-0.1, 0.1, 1)[0], np.random.uniform(-0.1, 0.1, 1)[0], filename=outfile)

res = np.loadtxt(outfile, delimiter=';', usecols=(0, 1))
print(res)
plt.scatter(res[:, 0], res[:, 1])
plt.show()
