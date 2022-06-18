#%% Load experiment data
from Toolbox.tools import mpath, evaluate2, CPRegressionResults
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from Toolbox.plot_helpers import barplot, scatter, compute_lim, finalize_plot
from sklearn.metrics import mean_squared_error

show_plots = True

pca = PCA()
results_path = mpath("results/protein_results/")


experiment_name = "Final_results_2"
path = mpath(results_path + experiment_name)
experiment_data = CPRegressionResults.load(path + "CP_data")

cp_results =     experiment_data.cp_results
data =           experiment_data.data
X_standard =     experiment_data.X_standard
y =              experiment_data.y
stratify =       experiment_data.stratify
train_X =        experiment_data.train_X
cal_X =          experiment_data.cal_X
test_X =         experiment_data.test_X
train_y =        experiment_data.train_y
cal_y =          experiment_data.cal_y
test_y =         experiment_data.test_y
train_strat =    experiment_data.train_strat
cal_strat =      experiment_data.cal_strat
test_strat =     experiment_data.test_strat


cp_result_group_names = set(map(lambda x: x.cp_model_name[:2], cp_results))
cp_result_groups = {group:[result for result in cp_results if result.cp_model_name[:2] == group] for group in cp_result_group_names}



# plot a bunch of stuff

plt.rcParams["figure.figsize"] = (6,4)
plt.rcParams["figure.dpi"] = 200

#%%

X_pca = pca.fit_transform(X_standard)
test_X_pca = pca.transform(test_X)

scatter(X_pca, y, alpha = 0.6)
plt.title("$y_{true}$ on PCs")
# plt.show()
finalize_plot(path = path + "PCA_of_X", show=show_plots)

#%%

group_dict = {'LR': 'Linear Regression',
              'RF': 'Random Forest'}

xlow = 0
xhigh = 25
for group_name, results in cp_result_groups.items():

    for result in results:
        print("")
        print(result.cp_model_name)
        print("empirical:                    ", result.empirical_coverage)
        print("mean prediction interval size:", np.mean(result.pred_set_sizes))
        print("min prediction interval size: ", np.min(result.pred_set_sizes))
        print("max prediction interval size: ", np.max(result.pred_set_sizes))
        print("mean effective sample size    ", result.mean_effective_sample_size)
        print("kernel outliers:              ", np.sum(result.kernel_errors))
        print("Model mean squared error:     ", mean_squared_error(test_y, result.y_preds))

        

    for result in results:
        if len(set(result.pred_set_sizes)) > 10:
            mask = (result.pred_set_sizes >= xlow)&(result.pred_set_sizes <= xhigh)
            plt.hist(result.pred_set_sizes[mask], density = True, label = result.cp_model_name[4:], alpha = 0.4, bins = 100)
        elif "Basic" in result.cp_model_name:
            plt.vlines(np.mean(result.pred_set_sizes), *plt.gca().get_ylim(), label = "Basic CP")
    plt.xlim(xlow,xhigh)
    plt.suptitle(group_dict[group_name])
    plt.legend()
    # plt.tight_layout()
    # plt.show()
    finalize_plot(path = path + "pred_set_size_histograms_" + group_dict[group_name], show=show_plots)
    
#%%

#Conditional coverage:

#Binned FSC

n_bins = 20
discard_frac = 0.1

for group_name, results in cp_result_groups.items():
    for i in range(4):
        min_x, max_x = np.quantile(test_X_pca[:,i], [discard_frac/2,1-discard_frac/2])
        mask = (min_x <= test_X_pca[:,i]) & (max_x >= test_X_pca[:,i])
        dx = (max_x-min_x)/n_bins
        x_labels = np.linspace(min_x + dx/2, max_x + dx/2, n_bins, endpoint=True)
        array = test_X_pca[mask,i]
        
        fig, ax1 = plt.subplots(dpi = 200, figsize = (7,5))
        ax2 = ax1.twinx()
        max_c =  0
        min_c =  float('inf')

        conditional_coverage_std = []
        for j, result in enumerate(results):
            conditional_coverage, count = evaluate2(array, result.in_pred_set, n_bins, min_x, max_x)
            pred_set_size, count = evaluate2(array, result.pred_set_sizes, n_bins, min_x, max_x)
            conditional_coverage_std.append(np.std(conditional_coverage))
            max_c =  max(max_c, np.max(conditional_coverage))
            min_c =  min(min_c, np.min(conditional_coverage))

            if "Basic" in result.cp_model_name:
                linewidth = 5
                alpha = 0.4
                color = 'black'
            else:
                linewidth = 1.5
                alpha = 1
                color = None

            ax1.plot(x_labels, conditional_coverage, label = f"{result.cp_model_name[4:]}, sd={conditional_coverage_std[-1]:.4f}",
                     color=color, linewidth=linewidth, alpha=alpha)
        ax2.plot(x_labels, count, label = "Bin Volume", color = 'r')

        ax2.set_ylim(0, np.max(count)*4)
        labels = ax2.get_yticks()
        ax2.set_yticks(labels[0::2]/4)

        ax1.set_ylim(min_c - (max_c-min_c)*0.4, max_c + (max_c-min_c)*0.5)
        plt.suptitle(group_dict[group_name])
        # labels = ax2.get_yticks()
        # ax2.set_yticks(labels/3)


        ax1.set_xlabel(f'PCA {i + 1}')
        ax1.set_ylabel('Empirical bin conditional coverage')
        ax2.set_ylabel('Number of data points')


        ax1.legend(loc = 'upper left')
        ax2.legend()
        finalize_plot(path = path + "binned_FSC_" + f"PC{i+1}_{group_dict[group_name]}", show=show_plots)

        
#%%
# Binned SSC



for group_name, results in cp_result_groups.items():
    min_x, max_x = float('inf'), -float('inf')
    for r in results:
        mi,ma = np.quantile(r.pred_set_sizes, [discard_frac/2,1-discard_frac/2])
        min_x = min(min_x, mi)
        max_x = max(max_x, ma)
    
    dx = (max_x-min_x)/n_bins
    x_labels = np.linspace(min_x + dx/2, max_x + dx/2, n_bins, endpoint=True)
    
    fig, ax1 = plt.subplots(dpi = 200, figsize = (7,5))
    max_c =  0
    min_c =  float('inf')

    conditional_coverage_std = []
    for j, result in enumerate(results):
        if "Basic" in result.cp_model_name: continue
        mask = (result.pred_set_sizes > min_x) & (result.pred_set_sizes < max_x)
        conditional_coverage, count = evaluate2(result.pred_set_sizes[mask], result.in_pred_set, n_bins, min_x, max_x)
        conditional_coverage_std.append(np.nanstd(conditional_coverage))
        max_c =  max(max_c, np.max(conditional_coverage))
        min_c =  min(min_c, np.min(conditional_coverage))


        linewidth = 1.5
        alpha = 1
        ax1.plot(x_labels, conditional_coverage, label = f"{result.cp_model_name[4:]}, sd={conditional_coverage_std[-1]:.4f}",
                 linewidth=linewidth, alpha=alpha)

    # for r in results:
        # if "Basic" in result.cp_model_name:
        #     plt.vlines(np.mean(result.pred_set_sizes), *plt.gca().get_ylim(), label = "Basic CP", color='black', linewidth = 6, alpha = 0.4)
    plt.suptitle(group_dict[group_name])

    ax1.set_xlabel(f'Prediction Set Size')
    ax1.set_ylabel('Empirical bin conditional coverage')


    ax1.legend(loc = 'upper left')
    ax2.legend()
    finalize_plot(path = path + "binned_SSC_" + f"{group_dict[group_name]}", show=show_plots)


#%%

plot_alpha = 0.6
s = 3
lim_level = 0.005

def replace(string, old_list, new):
    for c in old_list:
        string = string.replace(c,new)
    return string

for result in cp_results:
    plt.figure(figsize = (8,8), dpi = 300)
    ax1 = plt.subplot2grid((4, 5), (0, 0), rowspan=2, colspan=2)
    ax1.plot(
        test_y, 
        result.pred_set_sizes, 
        '.', markersize = s/2,
        alpha = plot_alpha
    )
    ax1.set_title("Prediction set set size vs $y_{true}$")
    ax1.set_xlabel("True label, $y_{true}$")
    ax1.set_ylabel(r"Prediction set size $|\hat C(x)|$")
    ax1.set_ylim(compute_lim(result.pred_set_sizes, lim_level))

    ax2 = plt.subplot2grid((4, 5), (2, 0), rowspan=2, colspan=2)
    ax2.plot(
        np.abs(test_y-result.y_preds), 
        result.pred_set_sizes, 
        '.', markersize=s/2,
        alpha = plot_alpha
    )
    ax2.set_title("Prediction set size vs true absolute difference")
    ax2.set_xlabel("$|y_{pred}-y_{true}$|")
    ax2.set_ylabel(r"Prediction set size $|\hat C(x)|$")
    xlim = ax2.get_xlim()
    if not "Basic" in result.cp_model_name:
        ax2.plot([0,100],[0,200], label="Prediction set border")
        ax2.legend()
    ax2.set_xlim(*xlim)
    ax2.set_ylim(compute_lim(result.pred_set_sizes, lim_level))
    # plt.show()
    # continue

    ax3 = plt.subplot2grid((4, 5), (0, 2), rowspan=2, colspan=3)
    quantity = result.pred_set_sizes
    scatter(test_X_pca, quantity, adapt_lim_level=lim_level, alpha=plot_alpha, s=s)
    ax3.set_title('pred set sizes on first two PCs')
    ax3.set_xlabel("PC 1")
    ax3.set_ylabel("PC 2")
    # ax3.plot()

    plt.tight_layout()

    ax4 = plt.subplot2grid((4, 5), (2, 2), rowspan=2, colspan=3)
    quantity = np.abs(result.y_preds - test_y)
    scatter(test_X_pca, quantity, adapt_lim_level=lim_level, alpha=plot_alpha, s=s)
    ax4.set_title('Absolute difference')
    ax4.set_xlabel("PC 1")
    ax4.set_ylabel("PC 2")
    ax4.plot()
    plt.suptitle(group_dict[result.cp_model_name[:2]] + result.cp_model_name[2:], fontsize = 15)

    finalize_plot(path = path + "overview_plots_" + replace(result.cp_model_name, r'.<>:"/\|?*', "_"), show=show_plots)







#%%

# export latex table

a = "{D^{test}}"
b = "|\hat C(x)|"

op = lambda s: f"$\\underset{a}{{\\text{{{s}}}}}{b}$"
# op = lambda s: s + f" ${b}$"

columns = {
           "Model":                       lambda r: r.cp_model_name[4:],
           "Empirical coverage":          lambda r: f"{r.empirical_coverage:.2%}",
        #    op('min'):   lambda r: f"{np.min(r.pred_set_sizes):.2f}",
        #    op('max'):   lambda r: f"{np.max(r.pred_set_sizes):.2f}",
           "Mean prediction set size":    lambda r: f"{np.mean(r.pred_set_sizes):.2f}",
        #    op('mean'):  lambda r: f"{np.mean(r.pred_set_sizes):.2f}",
           "Mean efffective sample size": lambda r: f"{r.mean_effective_sample_size:.2f}",
           }


# lines = ["\\begin{tabular}{" + f"p{{{1./len(columns):.2f}\\linewidth}}"*len(columns) + "}"]

lines = ["      " + " & ".join(columns.keys()) + "\\\\ \\hline"]

for group_name, results in cp_result_groups.items():

    for result in results:
        line = []
        for c,f in columns.items():
            line.append(f(result))
        lines.append("      " + " & ".join(line) + " \\\\")


    output = ("\n".join(lines) + " \\hline").replace("%", "\\%")

    print(output)



        