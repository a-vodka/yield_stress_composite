import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib import colors
import pandas as pd
import os


def plot_hist(data, value_name, units="", pdf=None, filename=None, kde=False):
    s = r'\left\langle{{ {0} }}\right\rangle = {1:.3f}\:\mathrm{{{2}}}'.format(value_name, np.mean(data), units)
    s += r';\quad'
    s += r'\sqrt{{\mathrm{{var}}\left[{0}\right]}}={1:.3f}\:\mathrm{{{2}}}'.format(value_name, np.std(data), units)
    s = '$' + s + '$'

    fig = plt.figure()
    ax = fig.add_subplot(111, title=s)

    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = ax.hist(data, density=True, bins=int(np.sqrt(data.size)), ec='black')

    # We'll color code by height, but you could use any scalar
    fracs = N / N.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    na_x = np.linspace(np.min(data), np.max(data), 1000)

    if pdf is not None:
        for dist_name in pdf:
            # Set up distribution and store distribution parameters
            dist = getattr(scipy.stats, str(dist_name))
            na_param = dist.fit(data, loc=np.mean(data), scale=np.std(data))
            ax.plot(na_x, dist.pdf(na_x, *na_param), label=dist_name)
        fig.legend(loc='center right')
    if kde:
        kernel = scipy.stats.gaussian_kde(data)
        ax.plot(na_x, kernel(na_x))

    x_lab = r'${0}$'.format(value_name)
    if units:
        x_lab += ", {}".format(units)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(r'$f({0})$'.format(value_name))
    fig.tight_layout()

    if filename:
        fig.savefig(filename + ".png", dpi=300)
        fig.savefig(filename)

    # print(scipy.stats.shapiro(data))


def filter(x, n_sigma=6):
    xm = np.mean(x)
    xs = np.std(x)
    cond = np.logical_and(x > xm - n_sigma * xs, x < xm + n_sigma * xs)
    return x[cond]


def pdf_fit(y_std):
    size = len(y_std)
    # Set list of distributions to test
    # See https://docs.scipy.org/doc/scipy/reference/stats.html for more

    # Turn off code warnings (this is not recommended for routine use)
    import warnings
    warnings.filterwarnings("ignore")

    # Set up list of candidate distributions to use
    # See https://docs.scipy.org/doc/scipy/reference/stats.html for more

    dist_names = [
        'expon',
        'gamma',
        'lognorm',
        'norm',
        'genextreme',
        'loggamma',
        'loglaplace',
        'gennorm',
        'exponnorm',
        #        'exponweib',

    ]

    # dist_names = ['alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'cauchy', 'chi', 'chi2',
    #               'cosine', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponweib', 'exponpow', 'f', 'fatiguelife',
    #               'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic', 'genpareto', 'genexpon',
    #               'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r',
    #               'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'invgamma', 'invgauss',
    #               'invweibull', 'johnsonsb', 'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'logistic', 'loggamma',
    #               'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm',
    #               'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh',
    #               'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda',
    #               'uniform', 'vonmises', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy']

    # dist_names = [
    #     "alpha", "anglit", "arcsine", "beta", "betaprime", "bradford", "burr", 'cauchy', 'chi', 'chi2',
    #     'cosine', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'fatiguelife',
    #     'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic', 'genpareto', 'gennorm', 'genexpon',
    #     'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r',
    #     'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'halfgennorm', 'hypsecant', 'invgamma',
    #     'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'levy', 'levy_l',
    #     'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami',
    #     'ncx2', 'ncf', 'nct', 'norm', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist',
    #     'reciprocal', 'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm',
    #     'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy'
    # ]

    dist_names = ['lognorm', 'exponnorm', 'norm']

    # Set up empty lists to stroe results
    chi_square = []
    p_values = []

    m1, m2, m3 = [], [], []

    # Set up 50 bins for chi-square test
    # Observed data will be approximately evenly distrubuted aross all bins
    percentile_bins = np.linspace(0, 100, 51)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)

    # Loop through candidate distributions

    for distribution in dist_names:
        # print(distribution)
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)

        m1.append(param[0])
        m2.append(param[1])
        if len(param) == 3:
            m3.append(param[2])
        else:
            m3.append(0)

        # Obtain the KS test P statistic, round it to 5 decimal places
        p = scipy.stats.kstest(y_std, distribution, args=param)[1]
        p = np.around(p, 5)
        p_values.append(p)

        # Get expected counts in percentile bins
        # This is based on a 'cumulative distrubution function' (cdf)
        cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2],
                              scale=param[-1])
        expected_frequency = []
        for bin in range(len(percentile_bins) - 1):
            expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # calculate chi-squared
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
        chi_square.append(ss)

    # Collate results and sort by goodness of fit (best at top)
    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square
    results['p_value'] = p_values
    results['m1'] = m1
    results['m2'] = m2
    results['m3'] = m3

    # results.sort_values(['chi_square'], inplace=True)

    results = results[results['p_value'] > 0.01]

    # Report results

    print('\nDistributions sorted by goodness of fit:')
    print('----------------------------------------')
    print(results)

    return results['Distribution'].to_numpy()


Exq = np.zeros([3, 18])
Eyq = np.zeros([3, 18])
Ezq = np.zeros([3, 18])
nuxy_q = np.zeros([3, 18])
nuxz_q = np.zeros([3, 18])
nuyz_q = np.zeros([3, 18])

Gyz_q = np.zeros([3, 18])
Gzx_q = np.zeros([3, 18])
Gxy_q = np.zeros([3, 18])


def main(r, i):
    fname = "./stress_out_tet/elastic_modules_{0:.2f}.csv".format(r)
    if not os.path.exists(fname):
        return
    data = np.loadtxt(fname, delimiter=';', dtype=float, comments='#')
    # data2 = np.loadtxt("./stress_out/principals_modules_{0}.csv".format(r), delimiter=';', dtype=float, comments='#')

    Ex = filter(data[:, 0] / 1e9)
    Ey = filter(data[:, 1] / 1e9)
    Ez = filter(data[:, 2] / 1e9)

    alpha = 0.9973
    quantiles = np.array([(1 - alpha) / 2, 0.5, (1 + alpha) / 2])
    Exq[:, i] = np.quantile(Ex, quantiles)
    Eyq[:, i] = np.quantile(Ey, quantiles)
    Ezq[:, i] = np.quantile(Ez, quantiles)

    # NUxy, NUxz, NUyz, Gyz, Gzx, Gxy
    nuxy = filter(data[:, 3])
    nuxz = filter(data[:, 4])
    nuyz = filter(data[:, 5])

    nuxy_q[:, i] = np.quantile(nuxy, quantiles)
    nuxz_q[:, i] = np.quantile(nuxz, quantiles)
    nuyz_q[:, i] = np.quantile(nuyz, quantiles)

    Gyz = filter(data[:, 6] / 1e9)
    Gzx = filter(data[:, 7] / 1e9)
    Gxy = filter(data[:, 8] / 1e9)

    Gyz_q[:, i] = np.quantile(Gyz, quantiles)
    Gzx_q[:, i] = np.quantile(Gzx, quantiles)
    Gxy_q[:, i] = np.quantile(Gxy, quantiles)

    # l1 = filter(data2[:, 0] / 1e9)
    # l2 = filter(data2[:, 1] / 1e9)
    # l3 = filter(data2[:, 2] / 1e9)
    # l4 = filter(data2[:, 3] / 1e9)
    # l5 = filter(data2[:, 4] / 1e9)
    # l6 = filter(data2[:, 5] / 1e9)

    plot_hist(Ex, "E_z, r = {0:.2f}".format(r), units="GPa", filename=None, pdf=pdf_fit(Ez)[0:3])
    # plot_hist(Ex, "E_x", units="GPa", filename="./stress_out_hist/ex_{0:.2f}.eps".format(r), pdf=pdf_fit(Ex)[0:3])
    # plot_hist(Ey, "E_y", units="GPa", filename="./stress_out_hist/ey_{0:.2f}.eps".format(r), pdf=pdf_fit(Ey)[0:3])
    # plot_hist(Ez, "E_z", units="GPa", filename="./stress_out_hist/ez_{0:.2f}.eps".format(r), pdf=pdf_fit(Ez)[0:3])
    #
    # plot_hist(nuxy, "\\nu_{xy}", units="", filename="./stress_out_hist/nuxy_{0:.2f}.eps".format(r),
    #           pdf=pdf_fit(nuxy)[0:3])
    # plot_hist(nuxz, "\\nu_{xz}", units="", filename="./stress_out_hist/nuxz_{0:.2f}.eps".format(r),
    #           pdf=pdf_fit(nuxz)[0:3])
    # plot_hist(nuyz, "\\nu_{yz}", units="", filename="./stress_out_hist/nuyz_{0:.2f}.eps".format(r),
    #           pdf=pdf_fit(nuyz)[0:3])
    #
    # plot_hist(Gxy, "G_{xy}", units="GPa", filename="./stress_out_hist/gxy_{0:.2f}.eps".format(r), pdf=pdf_fit(Gxy)[0:3])
    # plot_hist(Gzx, "G_{zx}", units="GPa", filename="./stress_out_hist/gzx_{0:.2f}.eps".format(r), pdf=pdf_fit(Gzx)[0:3])
    # plot_hist(Gyz, "G_{yz}", units="GPa", filename="./stress_out_hist/gyz_{0:.2f}.eps".format(r), pdf=pdf_fit(Gyz)[0:3])

    # plot_hist(l1, "\lambda_{1}", units="GPa", filename="./stress_out_hist/l1.eps", pdf=scipy.stats.norm)
    # plot_hist(l2, "\lambda_{2}", units="GPa", filename="./stress_out_hist/l2.eps", pdf=scipy.stats.norm)
    # plot_hist(l3, "\lambda_{3}", units="GPa", filename="./stress_out_hist/l3.eps", pdf=scipy.stats.norm)
    # plot_hist(l4, "\lambda_{4}", units="GPa", filename="./stress_out_hist/l4.eps", pdf=scipy.stats.norm)
    # plot_hist(l5, "\lambda_{5}", units="GPa", filename="./stress_out_hist/l5.eps", pdf=scipy.stats.norm)
    # plot_hist(l6, "\lambda_{6}", units="GPa", filename="./stress_out_hist/l6.eps", pdf=scipy.stats.norm)

    # plt.show()

    pass


if __name__ == "__main__":
    rad = np.linspace(0.1, 0.95, 18, endpoint=True)

    # rad = [0.9]
    for i in range(rad.size):
        main(rad[i], i)


    def rad_to_psi(r):
        return 0.5 * np.pi * r ** 2 / np.sqrt(3)


    def psi_to_rad(psi):
        return np.sqrt(psi / 0.5 * np.sqrt(3))


    filename = "./stress_out_hist/"
    fig, ax = plt.subplots(constrained_layout=True)
    fig.legend()
    ax.fill_between(rad, Exq[0, :], Exq[2, :], alpha=0.2, color='green')
    ax.plot(rad, Exq[1, :], 'g-')

    ax.fill_between(rad, Eyq[0, :], Eyq[2, :], alpha=0.2, color='red')
    ax.plot(rad, Eyq[1, :], 'r-')

    ax.fill_between(rad, Ezq[0, :], Ezq[2, :], alpha=0.2, color='blue')
    ax.plot(rad, Ezq[1, :], 'b-')
    ax.set_xlim([rad.min(), rad.max()])
    sec_axx = ax.secondary_xaxis('top', functions=(rad_to_psi, psi_to_rad))
    sec_axx.set_xlabel(r'$\psi$')
    ax.set_xlabel(r'fiber radius')
    ax.set_ylabel('Young\'s modulus, MPa')
    ax.grid()
    fig.savefig(filename + "E-psi.png", dpi=300)
    fig.savefig(filename + "E-psi.pdf")
    ##############################################################################
    fig, ax = plt.subplots(constrained_layout=True)
    ax.grid()
    fig.legend()
    ax.fill_between(rad, nuxy_q[0, :], nuxy_q[2, :], alpha=0.2, color='green')
    ax.plot(rad, nuxy_q[1, :], 'g-')

    ax.fill_between(rad, nuxz_q[0, :], nuxz_q[2, :], alpha=0.2, color='red')
    ax.plot(rad, nuxz_q[1, :], 'r-')

    ax.fill_between(rad, nuyz_q[0, :], nuyz_q[2, :], alpha=0.2, color='blue')
    ax.plot(rad, nuyz_q[1, :], 'b-')
    ax.set_xlim([rad.min(), rad.max()])

    sec_axx = ax.secondary_xaxis('top', functions=(rad_to_psi, psi_to_rad))
    sec_axx.set_xlabel(r'$\psi$')
    ax.set_xlabel(r'fiber radius')
    ax.set_ylabel('Poissons\'s ratios')
    fig.savefig(filename + "nu-psi.png", dpi=300)
    fig.savefig(filename + "nu-psi.pdf")

    ##############################################################################
    fig, ax = plt.subplots(constrained_layout=True)
    ax.grid()

    ax.fill_between(rad, Gxy_q[0, :], Gxy_q[2, :], alpha=0.2, color='green')
    ax.plot(rad, Gxy_q[1, :], 'g-', label='$G_{xy}$')

    ax.fill_between(rad, Gyz_q[0, :], Gyz_q[2, :], alpha=0.2, color='red')
    ax.plot(rad, Gyz_q[1, :], 'r-', label='$G_{yz}$')

    ax.fill_between(rad, Gzx_q[0, :], Gzx_q[2, :], alpha=0.2, color='blue')
    ax.plot(rad, Gzx_q[1, :], 'b-', label='$G_{zx}$')
    ax.set_xlim([rad.min(), rad.max()])

    sec_axx = ax.secondary_xaxis('top', functions=(rad_to_psi, psi_to_rad))
    sec_axx.set_xlabel(r'$\psi$')
    ax.set_xlabel(r'fiber radius')
    ax.set_ylabel('Shear modulus, MPa')
    fig.legend()
    fig.savefig(filename + "G-psi.png", dpi=300)
    fig.savefig(filename + "G-psi.pdf")

    plt.show()
