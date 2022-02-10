import numpy as np
from utility_K_exogenous_rate_endogenous_rate import *
# from utility_v5 import *
import pandas as pd


real_test = True

path_seq = ['sx-mathoverflow.txt']

for ii in range(len(path_seq)):
    path = path_seq[ii]
    name = path[1:(-4)]

    K = 5


    relations = (pd.read_csv(path, sep=" ", header=None).values).astype(float)
    relations = relations[:80000]

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
    origin_cov_f = 0
    origin_cov_g = 0
    origin_m_val = 0
    origin_points = 0

    origin_mu_g = 0
    origin_mu_f = 0

    setting_cov_val_f = [1.0, 0.5]
    setting_cov_val_g = [1.0, 0.5]

    # setting_cov_val_f = [3.0, 1]
    # setting_cov_val_g = [3.0, 1]

    adam_alpha = 0.05
    contec = CONTEC(adam_alpha, T, N, K, origin_pis, origin_cov_f, origin_cov_g, origin_m_val, origin_points, origin_mu_f, origin_mu_g, setting_cov_val_f, num_integration_points = 1000, num_inducing_points = 50, relations = relations, conv_crit=1e-4)
    # contec = CONTEC(T, N, K, origin_pis, origin_cov_f, origin_cov_g, origin_m_val, origin_points, origin_mu_f, origin_mu_g, setting_cov_val_f, setting_cov_val_g, num_integration_points = 1000, num_inducing_points = 50, nodepair = nodepair, eventtime = eventtime, conv_crit=1e-4)
    contec.num_iteration = 100
    print('formal start')

    elbo_previous = []
    elbo_seq = []
    ll_seq = []

    cov_para_ratio = []

    start_time = time.time()
    para_seq = []


    for ite in range(contec.num_iteration):

        contec.calculate_PG_expectation_f()
        contec.calculate_PG_expectation_g()
        if contec.var_evaluation['z_val']:
            contec.update_b_1D()
        if contec.var_evaluation['f_x']:
            contec.calculate_posterior_GP_f()
            contec.update_predictive_posterior_f()
        if contec.var_evaluation['g_x']:
            contec.calculate_posterior_GP_g()
            contec.update_predictive_posterior_g()
        if contec.var_evaluation['pis']:
            contec.update_pi_k()
        if contec.var_evaluation['m_val']:
            contec.update_m()


        contec.update_hyperparameters(ite+1)
        # ll_seq.append(contec.log_likelihood([0, contec.T]))


        elbo_seq.append(contec.calculate_lower_bound())

        # elbo_diff.append(elbo_seq[-1]-current_eblo)
        # if elbo_diff[-1]<0:
        #     contec.cov_params = current_cov_params
        #     contec.cov_params_g = current_cov_params_g
        #     contec.update_kernels()
        print('ELBO: ', elbo_seq[-1])
        print('Iteration: ', ite)

        # a = [contec.cov_params[k][1] for k in range(contec.K)]
        # para_seq.append(a)
        #
        #
        # cov_para_ratio.append([current_cov_params[k][1]/previous_cov_params[k][1] for k in range(contec.K)])
        # contec.cov_params = [copy.copy(setting_cov_val_f) for _ in range(K)]
        # contec.update_kernels()
        # elbo_origin.append(contec.calculate_lower_bound())
        # contec.cov_params = current_cov_params
        # contec.update_kernels()


        if (np.mod(ite, 20) == 0)&(ite>0):
            print('Elapsed time is: ', time.time() - start_time)
            print('iteration: ', ite + 1)
            start_time = time.time()
            plt.plot(elbo_seq)
            plt.savefig('elbo_'+name+'_K_exogenous_rate_endogenous_rate.pdf', edgecolor='none', format='pdf', bbox_inches='tight')
            plt.close()

    np.savez_compressed('result_'+name+'_K_exogenous_rate_endogenous_rate.npz', contec = contec, N = N, K = K, T = T, elbo_seq = elbo_seq)


# plt.figure(1)
# plt.subplot(221)
# plt.plot(elbo_seq, label = 'after-update')
# plt.plot(elbo_previous, label = 'previous')
# plt.legend()
# plt.subplot(222)
# plt.plot(elbo_diff)
#
# print('Cov-params-f: ', contec.cov_params)
# print('Cov-params-g: ', contec.cov_params_g)
# plt.subplot(223)
# cov_para_ratio = np.asarray(cov_para_ratio)
# plt.plot(cov_para_ratio[:, 0])
# plt.plot(cov_para_ratio[:, 1])
# para_seq = np.asarray(para_seq)
# plt.subplot(224)
# plt.plot(para_seq[:, 0])
# plt.plot(para_seq[:, 1])
# plt.ylim([0, 2])
# plt.show()
#
# w_val = np.sum(contec.z_ij_n_mat*contec.b_0_1D[:, np.newaxis], axis=0)
# print('Components ratio Z: ', w_val)
# a_val = (np.sum(np.exp(contec.log_pi_k), axis=0) ** 2) - np.sum(np.exp(contec.log_pi_k * 2), axis=0)
# print(r'Components ratio $\pi$: ', a_val)
#
# plt.savefig('Adam later Performance ' + str(adam_alpha) + '.pdf', edgecolor='none', format='pdf', bbox_inches='tight')
# plt.show()
#
# plt.figure(3)
# plt.subplot(121)
# plt.imshow(origin_pis)
# plt.subplot(122)
# plt.imshow(contec.pi_k)
# plt.show()
#
#
# compare_process(contec)
#
#
#
