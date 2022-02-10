import numpy as np
# from utility_v6 import *
# from utility_v5 import *
from utility_K_PoissonProcesses import *

np.random.seed(0)
N = 50
T = 5
K = 2

[relations, origin_pis, origin_cov_f, origin_cov_g, origin_m_val, origin_points, origin_mu_f, origin_mu_g] = inhomogeneous_poisson_process_sample(N, T, K)



# relations = origin_relations[:, :3]
print('Relation shape is: ', relations.shape)
ascending_index = np.argsort(relations[:, 2])
relations = relations[ascending_index]

nodepair = relations[:, :2].astype(int)
eventtime = relations[:, 2]
# cov_val = [1.0, 1.0]
setting_cov_val_f = [3.0, 1.]
setting_cov_val_g = [3.0, 1.]

# setting_cov_val_f = [3.0, 1]
# setting_cov_val_g = [3.0, 1]
# adam_alpha_seq = [0.5, 0.05, 0.005]
adam_alpha = 0.05
contec = CONTEC(adam_alpha, T, N, K, origin_pis, origin_cov_f, origin_points, origin_mu_f, setting_cov_val_f, num_integration_points = 1000, num_inducing_points = 50, relations = relations, conv_crit=1e-4)
# contec = CONTEC(T, N, K, origin_pis, origin_cov_f, origin_cov_g, origin_m_val, origin_points, origin_mu_f, origin_mu_g, setting_cov_val_f, setting_cov_val_g, num_integration_points = 1000, num_inducing_points = 50, nodepair = nodepair, eventtime = eventtime, conv_crit=1e-4)
contec.num_iteration = 50
print('formal start')


start_time = time.time()

for ite in range(contec.num_iteration):

    contec.calculate_PG_expectation_f()

    if contec.var_evaluation['f_x']:
        contec.calculate_posterior_GP_f()
        contec.update_predictive_posterior_f()

    if contec.var_evaluation['pis']:
        contec.update_pi_k()

    contec.update_z_ij_n()


    if np.mod(ite, 20) == 0:
        print('iteration: ', ite + 1)
        print('Elapse-time is: ', time.time()-start_time)
        start_time = time.time()


# plt.subplot()
# plt.plot(elbo_seq)
# plt.show()
#
# plt.subplot()


plt.figure(1)
eval_points = np.arange(0, T, 0.001)

f_seq = []
for kk in range(K):
    f_seq.append(gp_prediction(origin_points, origin_mu_f[kk], origin_cov_f[kk], eval_points))
f_seq = np.asarray(f_seq)
origin_val = sigma_f(np.asarray(f_seq))
a_val = (np.sum((origin_pis), axis=0) ** 2) - np.sum((origin_pis ** 2), axis=0)
ratioes = a_val / np.sum(a_val) * 100

plt.plot(eval_points, np.sum(a_val[:, np.newaxis] * origin_val / (contec.N * (contec.N - 1)), axis=0), alpha = 0.5, label = 'Groundtruth values')


mu_eval, variance_eval = contec.predictive_posterior_GP_f(eval_points)
val = sigma_f(np.asarray(mu_eval))
a_val = (np.sum((contec.pi_k), axis=0) ** 2) - np.sum((contec.pi_k ** 2), axis=0)
ratioes = a_val / np.sum(a_val) * 100
plt.plot(eval_points, np.sum(a_val[:, np.newaxis] * val / (contec.N * (contec.N - 1)), axis=0), alpha = 0.5, label = 'Fitted values')
plt.legend()
plt.title('Average exogenous functions')
plt.savefig('exogenous_rate_fitting_sim.pdf', edgecolor='none', format='pdf', bbox_inches='tight')
plt.close()
# plt.show()

plt.figure(figsize=(10, 2))
plt.subplot(211)
plt.imshow(origin_pis.T)
plt.title('Groundtruth values')
plt.axis('off')
plt.subplot(212)
plt.imshow(contec.pi_k.T)
plt.title('Fitted values')
plt.axis('off')
plt.savefig('latent_exogenous_feature_sim.pdf', edgecolor='none', format='pdf', bbox_inches='tight')
plt.close()
# plt.show()

# # plt.figure(1)
# plt.subplot(221)
# plt.plot(elbo_seq, label = 'after-update')
# plt.plot(elbo_previous, label = 'previous')
# plt.legend()
# plt.subplot(222)
# plt.plot(elbo_diff)
# plt.title('Adam alpha: '+str(adam_alpha))
# print('Cov-params-f: ', contec.cov_params)
# plt.subplot(223)
# cov_para_ratio = np.asarray(cov_para_ratio)
# plt.plot(cov_para_ratio[:, 0])
# plt.plot(cov_para_ratio[:, 1])
# para_seq = np.asarray(para_seq)
# plt.subplot(224)
# plt.plot(para_seq[:, 0])
# plt.plot(para_seq[:, 1])
#
# plt.savefig('Adam Performance '+str(adam_alpha)+'.pdf', edgecolor='none', format='pdf', bbox_inches='tight')
# plt.close()



