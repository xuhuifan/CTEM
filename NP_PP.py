import numpy as np
from utility_K_PoissonProcesses import *
# from utility_v5 import *

import pandas as pd
# np.random.seed(2)

simulation = False
real_test = True


path = 'sx-mathoverflow.txt'
name = path[1:(-4)]

K = 5
cov_params = [1., np.array([.01])]

relations = (pd.read_csv(path, sep=" ", header=None).values).astype(float)
relations = relations[:80000]
print(relations.shape)
relations = relations[(relations[:, 0] != relations[:, 1])]
unique_id = np.sort(np.unique(relations[:, :2]))
data_id = relations[:, :2]
for ii in range(len(unique_id)):
    data_id[data_id == unique_id[ii]] = ii
relations[:, 2] = (relations[:, 2] / (24 * 3600.0))
relations[:, 2] = relations[:, 2] - np.min(relations[:, 2])
T = np.max(relations[:, 2]) * 1.0001
N = len(unique_id)
origin_pis = np.ones((N, K))

lambda_star = 1
origin_cov_f = 0

origin_points = 0
origin_mu_f = 0



setting_cov_val_f = [1.0, 0.5]

adam_alpha = 0.05
contec = CONTEC(adam_alpha, T, N, K, origin_pis, origin_cov_f, origin_points, origin_mu_f, setting_cov_val_f, num_integration_points = 1000, num_inducing_points = 50, relations = relations, conv_crit=1e-4)
# contec = CONTEC(T, N, K, origin_pis, origin_cov_f, origin_cov_g, origin_m_val, origin_points, origin_mu_f, origin_mu_g, setting_cov_val_f, setting_cov_val_g, num_integration_points = 1000, num_inducing_points = 50, nodepair = nodepair, eventtime = eventtime, conv_crit=1e-4)
contec.num_iteration = 50
print('formal start')

elbo_previous = []
elbo_seq = []
elbo_diff = []

cov_para_ratio = []

start_time = time.time()
para_seq = []

for ite in range(contec.num_iteration):

    contec.calculate_PG_expectation_f()

    if contec.var_evaluation['f_x']:
        contec.calculate_posterior_GP_f()
        contec.update_predictive_posterior_f()

    if contec.var_evaluation['pis']:
        contec.update_pi_k()

    contec.update_z_ij_n()

    current_cov_params = copy.deepcopy(contec.cov_params)

    previous_cov_params = copy.deepcopy(contec.cov_params)

    current_eblo =(contec.calculate_lower_bound())
    elbo_previous.append(current_eblo)
    contec.update_hyperparameters(ite+1)

    elbo_seq.append(contec.calculate_lower_bound())

    print('iteration: ', ite + 1)

    if np.mod(ite, 20) == 0:
        print('Elapsed time is: ', time.time() - start_time)
        start_time = time.time()
        plt.plot(elbo_seq)
        plt.savefig('elbo_'+name+'_K_PoissonProcess.pdf', edgecolor='none', format='pdf', bbox_inches='tight')
        plt.close()
np.savez_compressed('result_' + name + '_K_PoissonProcess.npz', contec=contec, N=N, K=K, T=T, elbo_seq=elbo_seq)
