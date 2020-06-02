import numpy as np
import matplotlib.pyplot as plt
from pyAPDLWrapper.ansyswrapper import ansyswrapper
from numpy import linalg as LA
import os

fiber_radius = 0.9  # fiber radius

e_fiber = 74.8e9  # Pa
nu_fiber = 0.2  # Possion ratio
g_fiber = 0.5 * e_fiber / (1 + nu_fiber)  # Shear modulus, Pa
yield_fiber = 170e6  # Yeild stress, Pa

e_matrix = 2.7e9
nu_matirx = 0.4
g_matrix = 1.5e9
yield_matrix = 80e6


# e_fiber = e_matrix = 2.1e11
# nu_fiber = nu_matirx = 0.3
# g_fiber = g_matrix = e_fiber / 2 / (1 + nu_fiber)


def get_principals(tensor):
    s_principals, e_vectors = np.linalg.eig(tensor)
    a_max = np.argmax(np.abs(e_vectors), axis=1)
    s_principals = s_principals[a_max]
    return s_principals


def create_hexagonal_cell(ans, width, height):
    print("-----hexagonal_cell-----")
    s_fiber = 2 / 4 * np.pi * fiber_radius ** 2
    tot_area = width * height
    s_matrix = tot_area - s_fiber
    psi = s_fiber / tot_area
    print("psi = {0}".format(psi))

    E_mix = e_fiber * psi + e_matrix * (1 - psi)
    E_mix2 = (psi / e_fiber + (1 - psi) / e_matrix) ** -1

    yield_mix = yield_fiber * psi + yield_matrix * (1 - psi)
    yield_mix2 = (psi / yield_fiber + (1 - psi) / yield_matrix) ** -1

    print("Emix1 = {0:G}, \nEmix2 = {1:G},".format(E_mix, E_mix2))
    print("yield_mix1 = {0:G}, \nyield_mix2 = {1:G},".format(yield_mix, yield_mix2))

    ans.rectangle(0, 0, width, height)
    r1 = np.random.default_rng().normal(loc=1, scale=0.05, size=1) * fiber_radius
    r1[r1 > 1] = 0.98
    r1[fiber_radius ** 2 / r1 > 1] = fiber_radius ** 2 / 0.98
    r2 = fiber_radius ** 2 / r1
    print(r1, r2, "S = ", r1 * r2)

    ans.ellipse(0, 0, r1=r1[0], r2=r2[0])
    ans.ellipse(width, height, r1=r2[0], r2=r1[0])
    ans.overlapAreas()
    ans.delOuterArea(0, 0, width, height)
    print('-------------------------')
    return width, height


def create_tetragonal_cell(ans, width, height):
    print("-----tetragonal_cell-----")
    s_fiber = 1. / 4. * np.pi * fiber_radius ** 2
    tot_area = width * height
    s_matrix = tot_area - s_fiber
    psi = s_fiber / tot_area
    print("psi = {0}".format(psi))

    E_mix = e_fiber * psi + e_matrix * (1 - psi)
    E_mix2 = (psi / e_fiber + (1 - psi) / e_matrix) ** -1

    yield_mix = yield_fiber * psi + yield_matrix * (1 - psi)
    yield_mix2 = (psi / yield_fiber + (1 - psi) / yield_matrix) ** -1

    print("Emix = {0:G}, Emix2 = {1:G},".format(E_mix, E_mix2))
    print("yield_mix = {0:G}, yield_mix2 = {1:G},".format(yield_mix, yield_mix2))
    ans.rectangle(0, 0, width, height)

    r1 = np.random.default_rng().normal(loc=1, scale=0.05, size=2) * fiber_radius
    r2 = fiber_radius ** 2 / r1

    ans.ellipse(0, 0, r1=r1[0], r2=r2[0])
    ans.overlapAreas()
    ans.delOuterArea(0, 0, width, height)
    print('-------------------------')
    return width, height


def apply_ls_for_elastic_const_proc(ans):
    test_eps = 1e-5
    ans.applyLoadStep(0, 0, 1, np.sqrt(3), epsx=0, epsy=0)
    ans.applyLoadStep(0, 0, 1, np.sqrt(3), epsx=test_eps, epsy=0)
    ans.applyLoadStep(0, 0, 1, np.sqrt(3), epsx=0, epsy=test_eps)
    ans.applyLoadStep(0, 0, 1, np.sqrt(3), epsx=0, epsy=0, epsxy=test_eps)
    pass


def main(filename, fz):
    # psi = np.pi * fiber_radius ** 2 / cell_width / cell_height / 4.0
    # print("psi = {0}".format(psi))

    # E_mix = e_fiber * psi + e_matrix * (1 - psi)
    # E_mix2 = (psi / e_fiber + (1 - psi) / e_matrix) ** -1

    # yield_mix = yield_fiber * psi + yield_matrix * (1 - psi)
    # yield_mix2 = (psi / yield_fiber + (1 - psi) / yield_matrix) ** -1

    # print("Emix = {0:G}, Emix2 = {1:G},".format(E_mix, E_mix2))
    # print("yield_mix = {0:G}, yield_mix2 = {1:G},".format(yield_mix, yield_mix2))

    projdir = r'../ans_temp_dir'
    path_to_ans_bin = "/usr/ansys_inc/v201/ansys/bin/ansys201"

    ans = ansyswrapper(projdir=projdir, jobname='myjob', anslic='aa_t_i', path_to_and_bin=path_to_ans_bin,
                       isBatch=True)
    ans.setFEByNum(183)

    e_matrix_rnd = np.random.normal(1, 0.05, 1)[0] * e_matrix
    g_matrix_rnd = np.random.normal(1, 0.05, 1)[0] * g_matrix

    print('e_matrix_rnd = {0}'.format(e_matrix_rnd))

    # matrix_id = ans.createIsotropicMat(E=e_matrix_rnd, nu=nu_matirx)
    matrix_id = ans.createOrtotropicMat(Ex=e_matrix_rnd, Ey=e_matrix_rnd, Ez=e_matrix_rnd, nuxy=nu_matirx,
                                        nuxz=nu_matirx,
                                        nuyz=nu_matirx, GXY=g_matrix_rnd, GXZ=g_matrix_rnd, GYZ=g_matrix_rnd)
    fiber_id = ans.createIsotropicMat(E=e_fiber, nu=nu_fiber)

    ans.add_kxx_by_mat_id(mat_id=matrix_id, kxx=g_matrix_rnd)
    ans.add_kxx_by_mat_id(mat_id=fiber_id, kxx=g_fiber)

    # cell_width, cell_height = create_tetragonal_cell(ans, width=1.0, height=1.0)
    cell_width, cell_height = create_hexagonal_cell(ans, width=1, height=np.sqrt(3))

    ans.setCirlceAreaMatProps(x=0, y=0, rad=fiber_radius, matId=fiber_id)
    ans.setCirlceAreaMatProps(x=cell_width, y=cell_height, rad=fiber_radius, matId=fiber_id)

    ans.mesh(smartsize=1)

    # ans.applyTensX(0, 0, cell_width, cell_height)
    # ans.applyTensY(0, 0, cell_width, cell_height)
    # ans.applyTensXandY(0, 0, cell_width, cell_height)
    # ans.applyShearXY(0, 0, cell_width, cell_height, eps=ex)
    # ans.precessElasticConstants()

    if False:
        phi = np.linspace(0, 2 * np.pi, 100, dtype=np.float)

        rho = 1e-4
        ex = rho * np.cos(phi)
        ey = rho * np.sin(phi)
        for i in range(phi.size):
            ans.applyLoadStep(0, 0, 1, np.sqrt(3), epsx=ex[i], epsy=ey[i])

    # apply_ls_for_elastic_const_proc(ans=ans)

    test_eps = 1e-5
    ans.applyLoadStep(0, 0, cell_width, cell_height, epsx=0, epsy=0)
    ans.applyLoadStep(0, 0, cell_width, cell_height, epsx=test_eps, epsy=0)
    ans.applyLoadStep(0, 0, cell_width, cell_height, epsx=0, epsy=test_eps)
    ans.applyLoadStep(0, 0, cell_width, cell_height, epsx=0, epsy=0, epsxy=test_eps)

    # ans.solveAllLs()
    ans.solve_all_step_by_step(epsz=np.array([1e-5, 0, 0, 0]))
    ans.post()

    # ans.applyTensXandY(0, 0, cell_width, cell_height, epsx=ex, epsy=ey)
    ans.saveMaxStressForEachMaterial()

    ans.apdl += "/prep7\nETCHG,STT\n/SOL\n"
    ans.apply_thermal_ls_y(0, 0, cell_width, cell_height, temp=1e-3)
    ans.apply_thermal_ls_x(0, 0, cell_width, cell_height, temp=1e-3)
    ans.apdl += "LSSOLVE, {0}, {1}, 1,\n".format(ans.get_ls_num() - 2, ans.get_ls_num() - 1)

    ans.apdl += """
    ! Вход в постпроцессор
    /POST1
    temp_step = {3}
    SET,,,1,,temp_step
    PATH,line_Y_h,2,30,20,      !Создание пути интегрирования
    PPATH,1,,0,{1},0,0,  
    PPATH,2,,{0},{1},0,0, 
    PDEF,TFlY,TF,Y,AVG              
    PCALC,INTG,TFY_AV,TFLY,S,1,  !Интегрирование теплового потока по линии
    *GET,TFY_AV,PATH_INTEGRAL,0,LAST,TFY_AV ! Тепловой поток           ! С помощью PATH_INTEGRAL
    c55 = -TFY_AV/{2}*{1}
    SET,,,1,,temp_step+1
    PATH,line_X_h,2,30,20,      !Создание пути интегрирования
    PPATH,1,,{0},0,0,0,  
    PPATH,2,,{0},{1},0,0, 
    PDEF,TFlX,TF,X,AVG              
    PCALC,INTG,TFX_AV,TFLX,S,1,  !Интегрирование теплового потока по линии
    *GET,TFX_AV,PATH_INTEGRAL,0,LAST,TFX_AV ! Тепловой поток
    c66 = -TFX_AV/{2}/{1}
    
    *cfopen,c55c66,csv
    *vwrite,c55,c66
%G;%G
    *cfclose
    
    """.format(cell_width, cell_height, 0.001, ans.get_ls_num() - 2)

    ans.saveToFile(projdir + os.sep + 'test.apdl')

    ans.run()
    C = np.zeros([6, 6])

    sz, ez = ans.getAVGStressAndStrains(0)  # tension z
    sx, ex = ans.getAVGStressAndStrains(1)  # tension x
    sy, ey = ans.getAVGStressAndStrains(2)  # tension y
    sxy, exy = ans.getAVGStressAndStrains(3)  # tension xy

    GXY = sxy[0, 1] / exy[0, 1]
    res = np.loadtxt(projdir + os.sep + 'c55c66.csv', delimiter=';')

    ###
    ###
    ###
    # mat_s(1, 1) = SXX0
    # mat_s(1, 2) = SYY0
    # mat_s(1, 3) = SZZ0
    #
    # mat_s(2, 2) = SXX0
    # mat_s(2, 4) = SYY0
    # mat_s(2, 5) = SZZ0
    #
    # mat_s(3, 3) = SXX0
    # mat_s(3, 5) = SYY0
    # mat_s(3, 6) = SZZ0
    #
    # vec_b(1) = EXX0
    # vec_b(2) = 0
    # vec_b(3) = 0

    C[0, :] = np.array([sx[0, 0], sx[1, 1], sx[2, 2], 0, 0, 0])
    C[1, :] = np.array([0, sx[0, 0], 0, sx[1, 1], sx[2, 2], 0])
    C[2, :] = np.array([sy[0, 0], sy[1, 1], sy[2, 2], 0, 0, 0])
    #    C[2, :] = np.array([0, 0, sx[0, 0], 0, sx[1, 1], sx[2, 2]])
    C[3, :] = np.array([0, sy[0, 0], 0, sy[1, 1], sy[2, 2], 0])
    C[4, :] = np.array([0, sz[0, 0], 0, sz[1, 1], sz[2, 2], 0])
    # C[4, :] = np.array([0, 0, s[0, 0], 0, sy[1, 1], sy[2, 2]])
    C[5, :] = np.array([0, 0, sz[0, 0], 0, sz[1, 1], sz[2, 2]])

    b = np.array([ex[0, 0], 0, 0, ey[1, 1], 0, ez[2, 2]])
    S = np.linalg.solve(C, b)

    Ex = 1 / S[0]
    Ey = 1 / S[3]
    Ez = 1 / S[5]

    Gxy = GXY
    Gyz = res[0]
    Gzx = res[1]

    NUxy = -S[1] * Ex
    NUyx = -S[1] * Ey

    NUzx = -S[2] * Ez
    NUxz = -S[2] * Ex

    NUzy = -S[4] * Ez
    NUyz = -S[4] * Ey

    # print(C)
    # print(S)

    S_mat = np.array([[S[0], S[1], S[2], 0, 0, 0],
                      [S[1], S[3], S[4], 0, 0, 0],
                      [S[2], S[4], S[5], 0, 0, 0],
                      [0, 0, 0, 1. / Gyz, 0, 0],
                      [0, 0, 0, 0, 1. / Gzx, 0],
                      [0, 0, 0, 0, 0, 1. / Gxy],
                      ])

    C_mat = np.linalg.inv(S_mat)
    C_princ = get_principals(C_mat)
    # print(C_mat)

    print("Ex = {0:G}".format(Ex))
    print("Ey = {0:G}".format(Ey))
    print("Ez = {0:G}".format(Ez))

    print("Gxy = {0:G}".format(Gxy))
    print("Gyz = {0:G}".format(Gyz))
    print("Gxz = {0:G}".format(Gzx))

    print("Nuxy = ", NUxy)
    print("Nuyx = ", NUyx)

    print("Nuzx = ", NUzx)
    print("Nuxz = ", NUxz)

    print("Nuzy = ", NUzy)
    print("Nuyz = ", NUyz)

    save_array = np.array([Ex, Ey, Ez, NUxy, NUxz, NUyz, Gyz, Gzx, Gxy])
    out_file1 = open(
        os.path.dirname(os.path.realpath(__file__)) + "/stress_out/elastic_modules_{0}.csv".format(fiber_radius),
        mode='a')
    np.savetxt(out_file1, np.reshape(save_array, [1, save_array.size]), delimiter=';', newline='\n')
    out_file1.close()

    out_file2 = open(
        os.path.dirname(os.path.realpath(__file__)) + "/stress_out/principals_modules_{0}.csv".format(fiber_radius),
        mode='a')
    np.savetxt(out_file2, np.reshape(C_princ, [1, C_princ.size]), delimiter=';', newline='\n')
    out_file2.close()

    return
    ans.run()

    rnd_int = np.random.random_integers(100000)
    out_file = open(os.path.dirname(os.path.realpath(__file__)) + "/yield_out/yield{}.csv".format(rnd_int),
                    mode='a')
    for i in range(phi.size):
        s, e = ans.getAVGStressAndStrains(i)
        s_principals = get_principals(s)
        e_principals = get_principals(e)

        max_stress = ans.getMaxStressForEachMaterial(i)
        max1_stress = np.max(np.abs([max_stress[0, 1], max_stress[0, 3]]))
        max2_stress = np.max(np.abs([max_stress[1, 1], max_stress[1, 3]]))
        safety_factor = np.array([max1_stress / yield_matrix, max2_stress / yield_fiber])
        max_sf = np.max(safety_factor)

        yield_stress = s_principals / max_sf
        # yield_stress = [s[0, 0], s[1, 1], s[2, 2]] / np.min(maxs)

        print(yield_stress)
        print(s)
        print(s_principals)
        np.savetxt(out_file, np.reshape(yield_stress, (1, 3)), delimiter=';', newline='\n')

    out_file.close()

    pass


def plot_files():
    file_path = './yield_out/'
    files = os.listdir(file_path)

    i = 0
    N = len(files) - 1
    path_obj = np.empty(N, dtype=object)
    plt.axhline(y=0, lw=1, color='k')
    plt.axvline(x=0, lw=1, color='k')
    for f in files:
        if f.endswith('csv'):
            res = np.loadtxt(file_path + f, delimiter=';', usecols=[0, 1, 2])
            plt.plot(res[:, 0] / 1e6, res[:, 1] / 1e6)
            # plt.plot(res[:, 2], res[:, 3])
            # path_obj[i] = path.Path(res[:, 0:2])
            i += 1
    plt.xlabel(r"$\sigma_1$, MPa")
    plt.ylabel(r"$\sigma_2$, MPa")

    plt.tight_layout()
    plt.grid()
    plt.savefig('./stress_out/yield_spaghetti.png', dpi=300)
    plt.savefig('./stress_out/yield_spaghetti.eps')
    plt.savefig('./stress_out/yield_spaghetti.pdf')
    plt.show()
    plt.close()
    print("Loaded files = {}".format(i))
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for f in files:
        if f.endswith('csv'):
            res = np.loadtxt(file_path + f, delimiter=';', usecols=[0, 1, 2])
            ax.plot(res[:, 0] / 1e6, res[:, 1] / 1e6, res[:, 2] / 1e6)
    plt.show()


def elastic_main():
    rad = np.linspace(0.1, 0.9, 9, endpoint=True)+0.05
    for r in rad:
        global fiber_radius
        fiber_radius = r
        for i in range(500):
            main("", 0)


if __name__ == "__main__":
    elastic_main()

# outfile = "./yield_surface.csv"
# fz = np.linspace(-10, 10, 5) * 1e7
# for k in range(fz.size):
#    main(filename=outfile, fz=fz[k])
#    pass

# plot_files()
# res = np.loadtxt(outfile, delimiter=';', usecols=(0, 1))
# print(res)
# plt.scatter(res[:, 0], res[:, 1])
# plt.plot(res[:, 0], res[:, 1])
# plt.show()
