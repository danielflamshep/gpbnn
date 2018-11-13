import autograd.numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotting import sample_col
from bnn import sample_bnn, init_var_params
from gp import sample_gpp

def plot_samples(x_plot, p, title="", plot="bnn"):
    if plot =="bnn": col= "deep red"
    if plot =="gpp": col= "green"

    fig = plt.figure(facecolor='white')
    plt.plot(x_plot, p, sns.xkcd_rgb[col], lw=1)
    plt.tick_params(labelleft='off', labelbottom='off')
    plt.savefig(title+plot+"-samples.pdf", bbox_inches='tight')
    plt.clf()

def plot_density(funcs, plot="bnn"):
    f1, f2 = funcs[:, 250], funcs[:, 250 + 1]
    if plot=="bnn":
        sns.jointplot(f1, f2, kind='kde', color="xkcd:deep red")
    else:
        sns.jointplot(f1, f2, kind='kde', color="xkcd:green")

    plt.savefig("fspace" + str(250) +plot+".pdf", bbox_inches='tight')
    plt.clf()

def plot_save_priors_functions(bnn_arch=[1,20,1],bnn_act='rbf'):
    plot_inputs = np.linspace(-10, 10, num=500)[:, None]
    std_norm_param = init_var_params(bnn_arch, scale_mean=0, scale=1)
    f_bnn = sample_bnn(std_norm_param, plot_inputs, 3, bnn_arch, bnn_act)
    f_gps = sample_gpp(plot_inputs, 3)
    plot_samples(plot_inputs, f_bnn.T)
    plot_samples(plot_inputs, f_gps.T, plot="gpp")
    pass

def plot_save_priors_fdensity(bnn_arch=[1,20,1],bnn_act='rbf'):
    plot_inputs = np.linspace(-10, 10, num=500)[:, None]
    std_norm_param = init_var_params(bnn_arch, scale_mean=0, scale=1)
    f_bnn = sample_bnn(std_norm_param, plot_inputs, 25, bnn_arch, bnn_act)
    f_gps = sample_gpp(plot_inputs, 25)
    plot_density(f_bnn)
    plot_density(f_gps, plot="gpp")
    pass



if __name__ == '__main__':

    plot_save_priors_functions()
    plot_save_priors_fdensity()
