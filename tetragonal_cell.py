import numpy as np
import matplotlib.pyplot as plt
from pyAPDLWrapper.ansyswrapper import ansyswrapper
from numpy import linalg as LA


def main(ex, ey):
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

    psi = np.pi * fiber_radius ** 2 / cell_width / cell_height / 4.0
    # print("psi = {0}".format(psi))

    E_mix = e_fiber * psi + e_matrix * (1 - psi)
    E_mix2 = (psi / e_fiber + (1 - psi) / e_matrix) ** -1

    yield_mix = yield_fiber * psi + yield_matrix * (1 - psi)
    yield_mix2 = (psi / yield_fiber + (1 - psi) / yield_matrix) ** -1

    # print("Emix = {0:G}, Emix2 = {1:G},".format(E_mix, E_mix2))
    print("yield_mix = {0:G}, yield_mix2 = {1:G},".format(yield_mix, yield_mix2))
    exit()

    projdir = r'd:\ans_proj\compo_yield_stress'

    ans = ansyswrapper(projdir=projdir, jobname='myjob')
    ans.setFEByNum(183)

    matrix_id = ans.createIsotropicMat(E=e_matrix, nu=nu_matirx)
    fiber_id = ans.createIsotropicMat(E=e_fiber, nu=nu_fiber)

    ans.rectangle(0, 0, cell_width, cell_height)
    ans.circle(0, 0, fiber_radius)
    ans.overlapAreas()
    ans.delOuterArea(0, 0, cell_width, cell_height)
    ans.setCirlceAreaMatProps(fiber_radius, matId=fiber_id)
    ans.mesh()

    # ans.applyTensX(0, 0, cell_width, cell_height)
    # ans.applyTensY(0, 0, cell_width, cell_height)
    # ans.applyTensXandY(0, 0, cell_width, cell_height)
    # ans.applyShearXY(0, 0, cell_width, cell_height)
    # ans.precessElasticConstants()

    ans.applyTensXandY(0, 0, cell_width, cell_height, epsx=ex, epsy=ey)
    ans.saveMaxStressForEachMaterial()
    ans.saveToFile(projdir + '\\test.apdl')
    retcode = ans.run()
    if (not retcode):
        print('retcode = {0}'.format(retcode))
        exit(retcode)

    s, e = ans.getAVGStressAndStrains()
    s_principals, _ = LA.eig(s)

    maxs = ans.getMaxStressForEachMaterial() / np.array([yield_matrix, yield_fiber])

    # print(maxs)
    yield_stress = s_principals / np.min(maxs)

    print(yield_stress)

    out_file = open("yield_surface.csv", mode='a')
    np.savetxt(out_file, np.reshape(yield_stress, (1, 3)), delimiter=';', newline='\n')
    out_file.close()

    pass


ex = np.linspace(-0.1, 0.1, 5, endpoint=True)
ey = np.linspace(-0.1, 0.1, 5, endpoint=True)

xv, yv = np.meshgrid(ex, ey, sparse=False, indexing='ij')
for i in range(ex.size):
    for j in range(ey.size):
        print(xv[i, j], yv[i, j])
        main(xv[i, j], yv[i, j])
