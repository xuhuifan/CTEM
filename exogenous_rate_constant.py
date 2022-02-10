import numpy as np
from utility_constant_exogenous_rate import *
# from utility_v5 import *
import pandas as pd
# np.random.seed(2)

real_test = True

# path = '../dataset/email-Eu-core-temporal.txt'
path = '../dataset/CollegeMsg.txt'
# path = '../dataset/sx-mathoverflow.txt'
# path = '../dataset/sx-askubuntu.txt'
name = path[11:(-4)]

K = 5


relations = (pd.read_csv(path, sep=" ", header=None).values).astype(float)
relations = relations[:10000]

print('Data is: ', name)
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
origin_cov_g = 0
origin_m_val = 0
origin_points = 0

origin_mu_g = 0


# nodepair = relations[:, :2].astype(int)
# eventtime = relations[:, 2]

# cov_val = [1.0, 1.0]
setting_cov_val_g = [1.0, 0.5]

# setting_cov_val_f = [3.0, 1]
# setting_cov_val_g = [3.0, 1]

adam_alpha = 0.05
contec = CONTEC(adam_alpha, T, N, K, origin_pis, origin_cov_g, origin_m_val, origin_points, origin_mu_g, setting_cov_val_g, num_integration_points = 1000, num_inducing_points = 50, relations = relations, conv_crit=1e-4)
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

    contec.calculate_PG_expectation_g()

    if contec.var_evaluation['b_val']:
        contec.update_b_ij_n()

    if contec.var_evaluation['z_val']:
        contec.update_z_ij_n()

    if contec.var_evaluation['g_x']:
        contec.calculate_posterior_GP_g()
        contec.update_predictive_posterior_g()

    if contec.var_evaluation['m_val']:
        contec.update_m()
    if contec.var_evaluation['pis']:
        contec.update_pi_k()


    current_eblo =(contec.calculate_lower_bound())

    elbo_seq.append(current_eblo)

    # elbo_diff.append(elbo_seq[-1]-current_eblo)
    # if elbo_diff[-1]<0:
    #     contec.cov_params = current_cov_params
    #     contec.cov_params_g = current_cov_params_g
    #     contec.update_kernels()
    # print('ELBO: ', elbo_seq[-1])

    # a = [contec.cov_params[k][1] for k in range(contec.K)]
    # para_seq.append(a)


    if (np.mod(ite, 20) == 0)&(ite>0):
        print('Elapsed time is: ', time.time() - start_time)
        print('iteration: ', ite + 1)
        start_time = time.time()
        plt.plot(elbo_seq)
        plt.savefig('elbo_'+name+'_constant.pdf', edgecolor='none', format='pdf', bbox_inches='tight')
        plt.close()

np.savez_compressed('result_'+name+'_constant.npz', contec = contec, N = N, K = K, T = T, elbo_seq = elbo_seq)

