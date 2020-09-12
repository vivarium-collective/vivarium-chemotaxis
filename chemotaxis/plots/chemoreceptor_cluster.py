import os

from matplotlib import pyplot as plt


def plot_receptor_output(output, out_dir='out', filename='response'):
    ligand_vec = output['external']['MeAsp']  # TODO -- configure ligand name
    receptor_activity_vec = output['internal']['chemoreceptor_activity']
    n_methyl_vec = output['internal']['n_methyl']
    time_vec = output['time']

    # plot results
    cols = 1
    rows = 3
    plt.figure(figsize=(3.0 * cols, 2.5 * rows))
    plt.rc('font', size=12)

    ax1 = plt.subplot(rows, cols, 1)
    ax2 = plt.subplot(rows, cols, 2)
    ax3 = plt.subplot(rows, cols, 3)

    ax1.plot(time_vec, ligand_vec, 'steelblue')
    ax2.plot(time_vec, receptor_activity_vec, 'steelblue')
    ax3.plot(time_vec, n_methyl_vec, 'steelblue')

    ax1.set_xticklabels([])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.tick_params(right=False, top=False)
    ax1.set_ylabel("external ligand \n (mM) ", fontsize=10)
    # ax1.set_yscale('log')

    ax2.set_xticklabels([])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(right=False, top=False)
    ax2.set_ylabel("cluster activity \n P(on)", fontsize=10)

    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.tick_params(right=False, top=False)
    ax3.set_xlabel("time (s)", fontsize=12)
    ax3.set_ylabel("average \n methylation", fontsize=10)

    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(wspace=0.7, hspace=0.4)
    plt.savefig(fig_path + '.png', bbox_inches='tight')
    