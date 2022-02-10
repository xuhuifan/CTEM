import numpy as np
from scipy.stats import poisson, uniform, gamma, dirichlet, norm
from scipy.linalg import solve_triangular
from scipy.special import digamma, logsumexp, gammaln
import time
import copy
import matplotlib.pyplot as plt
import sys
import pyximport
pyximport.install()
from cython_function import judge_integration_relation, process_b_ij_n, gradient_g_part, z_part_2, PG_g_sum, PG_g_z, PG_g_m, PG_g_elbo
from scipy.integrate import quadrature



from sys import getsizeof, stderr
from itertools import chain
from collections import deque

def logcosh(x):
    # s always has real part >= 0
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + np.log1p(p) - np.log(2)


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

def sample_gp(x, cov_params):
    num_points = len(x)
    K = cov_func(x, x, cov_params)
    L = np.linalg.cholesky(K+1e-8 * np.eye(num_points))
    rand_nums = np.random.randn(num_points)

    return np.dot(L, rand_nums)

def compare_process(contec):
    # print(contec.origin_m_val)
    # print(np.exp(contec.log_m))
    # fig, ax = plt.subplots()
    plt.figure(4)
    test_points = np.linspace(0, contec.T, 2000)
    f_ori = []
    f_pre = []
    g_ori = []
    g_pre = []
    for kk in range(contec.K):
        plt.subplot(contec.K+1, 1, 1+kk)
        origin_val, origin_variance = gp_prediction_for_plot(contec.origin_points, contec.origin_mu_f[kk], contec.origin_cov_f[kk], test_points)
        predict_val, predict_variance = gp_prediction_for_plot(contec.induced_points, contec.f_s[kk], contec.cov_params[kk], test_points)
        # predict_val, predict_variance = gp_prediction_for_plot(contec.induced_points, norm.rvs(size=len(contec.f_s[kk])), contec.cov_params[kk], test_points)
        f_ori.append(origin_val)
        f_pre.append(predict_val)
        plt.plot(test_points, origin_val, label = 'Origin')
        plt.plot(test_points, predict_val, label = 'SCGP fitted')
        plt.fill_between(test_points, predict_val+predict_variance**(0.5), predict_val-predict_variance**(0.5), color='yellow', alpha = 0.5)
        # plt.plot([data, data], [-1, 1], c = 'red')
        plt.legend()
        plt.title('F val')

    predict_pi = np.sum(contec.pi_k, axis=0)**2-np.sum(contec.pi_k**2, axis=0)
    origin_pi = np.sum(contec.origin_pis, axis=0)**2-np.sum(contec.origin_pis**2, axis=0)
    plt.subplot(contec.K+1, 1, 1+(contec.K))
    plt.plot(test_points, (1/(1+np.exp(-np.asarray(f_ori).T))).dot(origin_pi), label = 'Origin')
    plt.plot(test_points, (1/(1+np.exp(-np.asarray(f_pre).T))).dot(predict_pi), label = 'SCGP fitted')
    # plt.plot([data, data], [-1, 1], c = 'red')
    # plt.legend()
    plt.title('F val')
    plt.savefig('F function later vis ' + str(contec.alpha_adam) + '.pdf', edgecolor='none', format='pdf', bbox_inches='tight')
    plt.show()

    # plt.figure(2)
    # for kk in range(contec.K):
    #     plt.subplot(contec.K, 2, 1+2*kk)
    #     origin_val = gp_prediction(contec.origin_points, contec.origin_mu_g[kk], contec.origin_cov_g[kk], test_points)
    #     predict_val = gp_prediction(contec.induced_points, contec.g_s[kk], contec.cov_params_g[kk], test_points)
    #     g_ori.append(origin_val)
    #     g_pre.append(predict_val)
    #     plt.plot(test_points, origin_val, label = 'Origin')
    #     plt.plot(test_points, predict_val, label = 'SCGP fitted')
    #     if kk ==0:
    #         plt.legend()
    #     plt.title('G val')
    #
    #     plt.subplot(contec.K, 2, 2+2*kk)
    #     plt.plot(test_points, contec.origin_m_val[kk]*(1/(1+np.exp(-np.asarray(origin_val)))), label = 'Origin')
    #     plt.plot(test_points, contec.m_k[kk]*(1/(1+np.exp(-np.asarray(predict_val)))), label = 'SCGP fitted')
    #
    #     plt.title('Sigmoid Cox')
    #
    #
    # # plt.subplot(contec.K+1, 1, 1+(contec.K))
    # # plt.plot(test_points, (1/(1+np.exp(-np.asarray(g_ori).T))).dot(contec.origin_m_val), label = 'Origin')
    # # plt.plot(test_points, (1/(1+np.exp(-np.asarray(g_pre).T))).dot(np.exp(contec.log_m)), label = 'SCGP fitted')
    # # # plt.legend()
    # # plt.title('G val')
    #
    #
    # plt.show()

def sigma_f(val_points):
    return (1+np.exp(-val_points))**(-1)


def cov_func(x, x_prime, cov_params):
    """ Computes the covariance functions between x and x_prime.

    :param x: np.ndarray [num_points x D]
        Contains coordinates for points of x
    :param x_prime: np.ndarray [num_points_prime x D]
        Contains coordinates for points of x_prime
    :param cov_params: list
        First entry is the amplitude, second an D-dimensional array with
        length scales.

    :return: np.ndarray [num_points x num_points_prime]
        Kernel matrix.
    """

    theta_1, theta_2 = cov_params[0], cov_params[1]

    x_theta2 = x[:, np.newaxis] / theta_2
    xprime_theta2 = x_prime[np.newaxis,:] / theta_2
    h = x_theta2** 2 - 2. * x_theta2*xprime_theta2 + xprime_theta2** 2
    return theta_1 * np.exp(-.5*h)

def gp_prediction_for_plot(origin_points, origin_mu, cov_params_k, test_points):
    noise = 0.01
    K_ori = cov_func(origin_points, origin_points, cov_params_k)
    L = np.linalg.cholesky(K_ori + noise * np.eye(K_ori.shape[0]))
    L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True, check_finite=False)

    K_ori_inv = L_inv.T.dot(L_inv)

    ks_x_prime = cov_func(origin_points, test_points, cov_params_k)
    kappa = K_ori_inv.dot(ks_x_prime)

    mu_test = kappa.T.dot(origin_mu)

    K_xx = cov_params_k[0] * np.ones(len(test_points))
    var_f_x_prime = K_xx - np.sum(kappa * (ks_x_prime - kappa.T.dot(K_ori).T), axis=0)

    return mu_test, var_f_x_prime


def gp_prediction(origin_points, origin_mu, cov_params_k, test_points):
    noise = 0.01
    K_ori = cov_func(origin_points, origin_points, cov_params_k)
    L = np.linalg.cholesky(K_ori + noise * np.eye(K_ori.shape[0]))
    L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True, check_finite=False)

    K_ori_inv = L_inv.T.dot(L_inv)

    ks_x_prime = cov_func(origin_points, test_points, cov_params_k)
    kappa = K_ori_inv.dot(ks_x_prime)

    mu_test = kappa.T.dot(origin_mu)

    return mu_test



def sample_inhomogeneous_poisson_process(maximum_val, T, start_point, function_type, origin_points, origin_mu_k, cov_params_k, delta_val):

    expected_nums = ((T-start_point) * maximum_val)
    actual_num_1 = poisson.rvs(mu=expected_nums)

    if actual_num_1>0:
        candidate_1 = uniform.rvs(size=actual_num_1) * (T-start_point)+start_point

        if function_type == 'GP':
            # try:
            f_1 = gp_prediction(origin_points, origin_mu_k, cov_params_k, candidate_1-start_point)
            # except:
            #     a = 1
            sigma_f_k = 1 / (1 + np.exp(-f_1))
        elif function_type == 'Expo':
            sigma_f_k = np.exp(-delta_val*(candidate_1-start_point))

        time_points = np.sort(candidate_1[uniform.rvs(size=actual_num_1) < sigma_f_k])
        return_val = time_points.tolist()
    else:
        return_val = []

    return return_val

def inhomogeneous_poisson_process_sample(N, T, K):
    pi_parameter = gamma.rvs(a=2, size=(2, K), scale = 0.1)

    cov_params_f = [[3., 1.] for _ in range(K)]

    cov_params_g = [[3., 1.] for _ in range(K)]

    pis = gamma.rvs(a = 10*pi_parameter[0][np.newaxis, :], size = (N,K), scale = 0.1)
    m_val = np.asarray([0.5, 0.2, 0.6, 0.3, 2, 0.1, 0.35, 0.6, 0.4, 0.4])[:K]
    origin_points = np.linspace(0, T, 50)
    origin_mu_f = [np.random.randn(len(origin_points)) for _ in range(K)]
    # origin_mu_g = [np.sort(np.random.randn(len(origin_points)))[::-1] for _ in range(K)]
    origin_mu_g = [(np.random.randn(len(origin_points))) for _ in range(K)]

    relations = []
    for i in range(N):
        for j in np.arange(i+1, N).astype(int):
            # print('i: ', i, ', j: ', j)
            time_1 = []
            time_2 = []
            start_point = 0
            function_type = 'GP'
            pointz_ij = []
            for k in range(K):
                maximum_val = pis[i][k]*pis[j][k]
                origin_mu_k = origin_mu_f[k]
                cov_params_k = cov_params_f[k]
                delta_val = []
                time_1_k = (sample_inhomogeneous_poisson_process(maximum_val, T, 0, function_type, origin_points, origin_mu_k,
                                                     cov_params_k, delta_val))
                time_1.append(time_1_k)

                time_2_k = (sample_inhomogeneous_poisson_process(maximum_val, T, 0, function_type, origin_points, origin_mu_k,
                                                     cov_params_k, delta_val))
                time_2.append(time_2_k)

            for k in range(K):
                existing_points_1 = (time_1[k])
                existing_points_2 = (time_2[k])

                maximum_val = m_val[k]
                origin_mu_k = origin_mu_g[k]
                cov_params_k = cov_params_g[k]
                delta_val = []
                while (len(existing_points_1)>0)|(len(existing_points_2)>0):
                    if (len(existing_points_1)==0):
                        start_point_1 = T + 0.1
                    else:
                        start_point_1 = existing_points_1[0]
                    if (len(existing_points_2)==0):
                        start_point_2 = T + 0.1
                    else:
                        start_point_2 = existing_points_2[0]
                    if start_point_1<start_point_2:
                        time_stand = start_point_1
                        exiting_point = [i,j, time_stand, k]
                        # existing_points_2.extend(sample_inhomogeneous_poisson_process(maximum_val, T, time_stand, function_type, origin_points,
                        #                                      origin_mu_k,cov_params_k, delta_val))
                        existing_points_1.pop(0)
                        # print('points_1 len: ', len(existing_points_1))
                        existing_points_2 = np.sort(existing_points_2).tolist()
                    else:
                        time_stand = start_point_2
                        exiting_point = [j,i, time_stand, k]
                        # existing_points_1.extend(sample_inhomogeneous_poisson_process(maximum_val, T, time_stand, function_type, origin_points,
                        #                                      origin_mu_k,cov_params_k, delta_val))
                        existing_points_2.pop(0)
                        # print('points_2 len: ', len(existing_points_2))
                        existing_points_1 = np.sort(existing_points_1).tolist()
                    # print('elapsed time: ', time_stand)
                    pointz_ij.append(exiting_point)
                # print('finish k: ', k)

            relations.extend(pointz_ij)
    relations = np.asarray(relations)

    return relations, pis, cov_params_f, cov_params_g, m_val, origin_points, origin_mu_f, origin_mu_g


class CONTEC():
    def __init__(self, adam_alpha, T, N, K, pis, origin_cov_f, origin_points, origin_mu_f, cov_params, num_integration_points, num_inducing_points, relations, conv_crit=1e-4):
        self.update_hyper_f = True

        # ADAM parameters
        self.alpha_adam = adam_alpha

        self.beta1_adam = .9
        self.beta2_adam = .999
        self.epsilon_adam = 1e-5


        self.T = T
        self.K = K
        self.N = N
        self.cov_params = [copy.copy(cov_params) for _ in range(K)]

        self.num_integration_points = num_integration_points
        self.num_inducing_points = num_inducing_points
        self.relations = relations
        self.NumE = relations.shape[0]
        self.noise = 0.001

        self.origin_pis = pis

        self.num_iteration = 100


        self.log_pi_k = np.log(np.ones(pis.shape))
        self.pi_k = np.exp(self.log_pi_k)


        ##############################################################################
        ##############################################################################

        var_evaluation = {'pis': True,
                          'f_x': True
                          }


        self.var_evaluation = var_evaluation

        self.init_sb()

        self.place_inducing_points()


        ##############################################################################
        ##############################################################################


        self.f_s = [norm.rvs(loc = 0, scale = 1, size=self.induced_points.shape[0]) for _ in range(K)]
        self.Sigma_f_s = [np.identity(self.induced_points.shape[0]) for _ in range(K)]
        self.Sigma_f_s_inv = [np.identity(self.induced_points.shape[0]) for _ in range(K)]


        ##############################################################################
        ##############################################################################


        self.Ks = [cov_func(self.induced_points, self.induced_points, self.cov_params[k]) for k in range(K)]
        L = [np.linalg.cholesky(self.Ks[k] + self.noise*np.eye(self.Ks[k].shape[0])) for k in range(K)]
        L_inv = [solve_triangular(L[k], np.eye(L[k].shape[0]), lower = True, check_finite=False) for k in range(K)]
        self.Ks_inv = [L_inv[k].T.dot(L_inv[k]) for k in range(K)]
        self.logdet_Ks = [2. * np.sum(np.log(L[k].diagonal())) for k in range(self.K)]


        ##############################################################################
        ##############################################################################


        # self.logdet_Ks = [2.*np.sum(np.log(L[k].diagonal())) for k in range(K)]

        self.place_integration_points()

        self.origin_cov_f = origin_cov_f

        self.origin_points = origin_points
        self.origin_mu_f = origin_mu_f


        [judge_per_relation, judge_per_inte_points] = judge_integration_relation(self.relations[:, 2], self.integration_points, \
                                                                                 np.intc(self.NumE), np.intc(self.num_integration_points), self.T)
        self.judge_per_relation = np.asarray(judge_per_relation)
        self.judge_per_inte_points = np.asarray(judge_per_inte_points)
        # print('diff 1: ', np.sum(abs(self.judge_per_relation-np.asarray(judge_per_relation))))
        # print('diff 2: ', np.sum(abs(self.judge_per_inte_points-np.asarray(judge_per_inte_points))))

        ##############################################################################
        ##############################################################################


        self.ks_X = [cov_func(self.induced_points, relations[:, 2], self.cov_params[k]) for k in range(K)]
        self.kappa_X = [self.Ks_inv[k].dot(self.ks_X[k]) for k in range(K)]

        self.f_X, var_f_X = self.predictive_posterior_GP_f(self.relations, 'X')
        self.f2_X = [var_f_X[k] + self.f_X[k]**2 for k in range(K)]
        self.f_int_points, var_f_int_points = self.predictive_posterior_GP_f(self.integration_points, 'int_points')
        self.f2_int_points = [var_f_int_points[k] + self.f_int_points[k]**2 for k in range(K)]


        self.z_ij_n_mat = np.ones((relations.shape[0], self.K))

        ##############################################################################
        ##############################################################################


        self.alpha_pi_0 = 0.01*np.ones(self.N)
        self.beta_pi_0 = 0.1*np.ones(self.N)

        self.N_judge_index = []
        for ii in range(self.N):
            first_judge = (self.relations[:, 0]==ii)
            second_judge = (self.relations[:, 1]==ii)
            self.N_judge_index.append(np.where(np.logical_or(first_judge, second_judge))[0])


        if not self.var_evaluation['pis']:
            nor = pis + np.random.rand(len(pis))
            self.log_pi_k = np.log(nor/np.sum(nor))

        if not self.var_evaluation['f_x']:
            self.f_s = [gp_prediction(self.origin_points, self.origin_mu_f[k], self.origin_cov_f[k], self.induced_points) for k in range(self.K)]
            self.f_X = [gp_prediction(self.origin_points, self.origin_mu_f[k], self.origin_cov_f[k], self.relations[:, 2]) for k in range(self.K)]
            self.f2_X = [self.f_X[k]**2 for k in range(K)]
            self.f_int_points= [gp_prediction(self.origin_points, self.origin_mu_f[k], self.origin_cov_f[k], self.integration_points) for k in range(self.K)]
            self.f2_int_points = [self.f_int_points[k]**2 for k in range(K)]


        ##############################################################################
        ##############################################################################


        self.m_hyper_adam = np.zeros((self.K, 1+ 1))
        self.v_hyper_adam = np.zeros((self.K, 1+ 1))

        self.m_bm_adam = np.zeros(1)
        self.v_bm_adam = np.zeros(1)


    def place_inducing_points(self):
        """ Places the induced points for sparse GP.
        """

        num_per_dim = int(np.ceil(self.num_inducing_points))
        dist_between_points = self.T/num_per_dim
        induced_grid = np.arange(.5*dist_between_points,self.T,dist_between_points)

        self.induced_points = induced_grid

    def place_integration_points(self):
        """ Places the integration points for Monte Carlo integration and
        updates all related kernels.
        """

        self.integration_points = (np.random.rand(self.num_integration_points))
        self.integration_points *= self.T
        self.integration_points = np.concatenate((self.integration_points, [0.5*(self.T-self.relations[-1, 2])]))
        self.integration_points = np.sort(self.integration_points)
        self.num_integration_points += 1
        self.ks_int_points = [cov_func(self.induced_points,self.integration_points, self.cov_params[k]) for k in range(self.K)]
        self.kappa_int_points = [self.Ks_inv[k].dot(self.ks_int_points[k]) for k in range(self.K)]



    def init_sb(self):

        unique_pair = np.unique(self.relations[:, :2], axis=0).astype(int)
        pair_index = []
        for pair_i in range(len(unique_pair)):
            judge = np.where(np.sum(self.relations[:, :2]==unique_pair[pair_i][::-1], axis=1)==2)[0]
            pair_index.append(judge)


        ##############################################################################
        ##############################################################################


        self.mutually_before = []
        self.diff_X_before = []
        mutually_after = []
        diff_X_after = []

        for ii in range(self.NumE):
            judge1 = np.where(np.sum(self.relations[ii, :2]==unique_pair, axis=1)==2)[0]
            judge = pair_index[judge1[0]]
            if len(judge) == 0:
                self.mutually_before.append(np.asarray([]))
                mutually_after.append(np.asarray([]))

                diff_X_after.append([])
                self.diff_X_before.append([])

            else:
                self.mutually_before.append(judge[self.relations[judge,2]<=self.relations[ii, 2]])
                mutually_after.append(judge[self.relations[judge,2]>self.relations[ii, 2]])

                ii_time_after = self.relations[mutually_after[ii], 2] - self.relations[ii, 2]
                ii_time_before = self.relations[ii, 2] - self.relations[self.mutually_before[ii], 2]
                diff_X_after.append(ii_time_after)
                self.diff_X_before.append(ii_time_before)


        self.diff_before_len = [len(diff_before_i) for diff_before_i in self.diff_X_before]
        self.before_max = np.intc(max(self.diff_before_len)+1)
        diff_before_len_cum = np.cumsum(self.diff_before_len)
        self.diff_before_len_cum = np.concatenate([np.array([0]), diff_before_len_cum])
        self.diff_X_before_1D = np.concatenate(self.diff_X_before, axis=0)
        self.mutually_before_1D = np.concatenate(self.mutually_before, axis=0).astype(int)

        self.diff_after_len = np.asarray([len(diff_after_i) for diff_after_i in diff_X_after])
        diff_after_len_cum = np.cumsum(self.diff_after_len)
        self.diff_after_len_cum = np.concatenate([np.array([0]), diff_after_len_cum])
        diff_X_after_1D = np.concatenate(diff_X_after, axis=0)


        ##############################################################################
        ##############################################################################


        nums = np.arange(len(self.diff_X_before_1D))
        nums_list = [(nums[self.diff_before_len_cum[ii]:self.diff_before_len_cum[ii+1]]) for ii in range(self.NumE)]

        ji_n_slash_index = []
        for ii in range(self.NumE):
            if len(mutually_after[ii])>0:
                for jj in range(len(mutually_after[ii])):
                    jj_index = (np.where(self.mutually_before[mutually_after[ii][jj]]==ii)[0][0])
                    mutually_ii_jj = mutually_after[ii][jj]
                    ji_n_slash_index.append(nums_list[mutually_ii_jj][jj_index])


        self.ji_n_slash_index = np.asarray(ji_n_slash_index)
        # print(np.sum(abs(diff_X_after_1D-self.diff_X_before_1D[ji_n_slash_index])))

        ##############################################################################
        ##############################################################################

        # self.z_ij_n_mat = dirichlet.rvs(alpha = np.ones(self.K), size=self.NumE)
        self.z_ij_n_mat = np.ones((self.NumE, self.K))/self.K




    def calculate_PG_expectation_f(self):

        self.c_X_f = np.sqrt(np.asarray(self.f2_X)*self.z_ij_n_mat.T)
        self.c_int_points_f = np.sqrt(self.f2_int_points)

        self.w_X_f = 0.5*np.tanh(0.5*self.c_X_f)/self.c_X_f
        self.w_int_points_f = 0.5*np.tanh(0.5*self.c_int_points_f)/self.c_int_points_f


        self.Lambda_f_without_pik = 0.5*np.exp(-0.5*np.asarray(self.f_int_points))/np.cosh(0.5*self.c_int_points_f)




    def update_z_ij_n(self):

        mu_alpha_ij = (self.log_pi_k[(self.relations[:, 0]).astype(int)]+self.log_pi_k[(self.relations[:, 1]).astype(int)])

        first_part = mu_alpha_ij + (np.asarray(self.f_X)/2-np.asarray(self.f2_X)*self.w_X_f/2-np.log(2)).T

        nor_ll = first_part-logsumexp(first_part, axis=1)[:, np.newaxis]

        z_ij_n_mat = np.exp(nor_ll)

        self.z_ij_n_mat = (z_ij_n_mat+1e-10)/(1+self.K*1e-10)

        # print(self.z_ij_n_mat[:10])


    def calculate_posterior_GP_f(self):

        a_val = (np.sum(np.exp(self.log_pi_k), axis=0)**2)-np.sum(np.exp(self.log_pi_k*2), axis=0)

        Lambda_f_sum = self.Lambda_f_without_pik*(a_val[:, np.newaxis])
        A_int_points = self.w_int_points_f*Lambda_f_sum
        B_int_points = -0.5*Lambda_f_sum

        # A_X = (self.w_X_f)
        # B_X = self.relations.shape[0]*0.5

        A_X = (self.w_X_f*(self.z_ij_n_mat.T))
        B_X = (self.z_ij_n_mat.T)*0.5


        ##############################################################################
        ##############################################################################


        kAk = [self.kappa_X[k].dot(A_X[k][:, np.newaxis]*self.kappa_X[k].T) + \
            self.kappa_int_points[k].dot(A_int_points[k][:, np.newaxis]*self.kappa_int_points[k].T) \
            /self.num_integration_points*self.T for k in range(self.K)]
        self.Sigma_f_s_inv = [kAk[k]+self.Ks_inv[k] for k in range(self.K)]
        L_inv = [np.linalg.cholesky(self.Sigma_f_s_inv[k]+self.noise*np.eye(self.Sigma_f_s_inv[k].shape[0])) for k in range(self.K)]
        L = [solve_triangular(L_inv[k], np.eye(L_inv[k].shape[0]), lower=True, check_finite=False) for k in range(self.K)]
        self.Sigma_f_s = [L[k].T.dot(L[k]) for k in range(self.K)]
        self.logdet_Sigma_f_s = [2*np.sum(np.log(L[k].diagonal())) for k in range(self.K)]


        ##############################################################################
        ##############################################################################



        Kb = [self.ks_X[k].dot(B_X[k])+(self.ks_int_points[k].dot(B_int_points[k])/self.num_integration_points*self.T) for k in range(self.K)]

        self.f_s = [self.Sigma_f_s[k].dot(Kb[k].dot(self.Ks_inv[k])) for k in range(self.K)]



    def update_predictive_posterior_f(self, only_int_points = False):
        if not only_int_points:
            mu_f_X, var_f_X = self.predictive_posterior_GP_f(self.relations, points = 'X')
            self.f_X = mu_f_X
            self.f2_X = [var_f_X[k] + mu_f_X[k]**2 for k in range(self.K)]

        mu_f_int_points, var_f_int_points = self.predictive_posterior_GP_f(self.integration_points, \
                                                                         points = 'int_points')
        self.f_int_points = mu_f_int_points
        self.f2_int_points = [var_f_int_points[k] + mu_f_int_points[k]**2 for k in range(self.K)]


    def predictive_posterior_GP_f(self, x_prime, points=None):
        if points is None:
            ks_x_prime = [cov_func(self.induced_points, x_prime, self.cov_params[k]) for k in range(self.K)]
            kappa = [self.Ks_inv[k].dot(ks_x_prime[k]) for k in range(self.K)]
        elif points is 'int_points':
            ks_x_prime = self.ks_int_points
            kappa = self.kappa_int_points
        elif points is 'X':
            ks_x_prime = self.ks_X
            kappa = self.kappa_X

        mu_f_x_prime = [kappa[k].T.dot(self.f_s[k]) for k in range(self.K)]
        K_xx = [self.cov_params[k][0]*np.ones(x_prime.shape[0]) for k in range(self.K)]
        var_f_x_prime = [K_xx[k] - np.sum(kappa[k]*(ks_x_prime[k] - kappa[k].T.dot(self.Sigma_f_s[k]).T),axis=0) for k in range(self.K)]

        return mu_f_x_prime, var_f_x_prime



    def update_pi_k(self):
        self.alpha_pi_q = np.zeros((self.N, self.K))
        self.beta_pi_q = np.zeros((self.N, self.K))

        sum_exp_log_pi_k = np.sum(np.exp(self.log_pi_k), axis=0)
        sum_pi_k = np.sum(self.pi_k, axis=0)
        for ii in range(self.N):

            part_1 = np.sum(self.z_ij_n_mat[self.N_judge_index[ii]], axis=0)

            a_val = np.exp(self.log_pi_k[ii])*2 * (sum_exp_log_pi_k-np.exp(self.log_pi_k[ii]))

            b_val = self.Lambda_f_without_pik * a_val[:, np.newaxis]

            self.alpha_pi_q[ii] = np.sum(b_val, axis=1)/self.num_integration_points*self.T+part_1+self.alpha_pi_0[ii]
            c_val = 2*(sum_pi_k-self.pi_k[ii])
            self.beta_pi_q[ii] = self.T*c_val + self.beta_pi_0[ii]

        self.log_pi_k = digamma(self.alpha_pi_q)-np.log(self.beta_pi_q)
        self.pi_k = self.alpha_pi_q/self.beta_pi_q

    def log_likelihood(self, time_period):
        """ Calculates the variational lower bound for current posterior.

        :return: float
            Variational lower bound.
        """
        a_val = (np.sum(np.exp(self.log_pi_k), axis=0)**2)-np.sum(np.exp(self.log_pi_k*2), axis=0)
        eval_points = np.arange(time_period[0], time_period[1], step = 0.001)
        eval_f, _ = self.predictive_posterior_GP_f(eval_points)

        sigma_f_val = [sigma_f(eval_f_i) for eval_f_i in eval_f]
        integration_ij = [sum(sigma_f_val[kk])*self.T/len(eval_points) for kk in range(self.K)]
        integration_f = -sum([integration_ij[kk]*a_val[kk] for kk in range(self.K)])

        index_m = np.where((self.relations[:,2]>=time_period[0])&(self.relations[:,2]<time_period[1]))[0]
        log_val = self.log_pi_k[self.relations[index_m, 0].astype(int)]+self.log_pi_k[self.relations[index_m, 1].astype(int)]
        sum_f = np.sum(self.z_ij_n_mat[index_m]*(log_val + np.log(sigma_f((np.asarray(self.f_X).T)[index_m]))))

        return integration_f + sum_f


    def log_likelihood_future(self, time_period):
        """ Calculates the variational lower bound for current posterior.

        :return: float
            Variational lower bound.
        """
        # a_val = (np.sum(np.exp(self.log_pi_k), axis=0)**2)-np.sum(np.exp(self.log_pi_k*2), axis=0)
        eval_points = np.arange(time_period[0], time_period[1], step = 0.001)
        eval_f, _ = self.predictive_posterior_GP_f(eval_points)

        sigma_f_val = [sigma_f(eval_f_i) for eval_f_i in eval_f]
        integration_ij = [sum(sigma_f_val[kk])*(time_period[1]-time_period[0])/len(eval_points) for kk in range(self.K)]

        pi_f = np.exp(self.log_pi_k)[:, np.newaxis, :]*np.exp(self.log_pi_k)[np.newaxis, :, ]
        integration_f = (np.sum(pi_f*(np.asarray(integration_ij)[np.newaxis, np.newaxis, :]), axis=2))

        return integration_f



    def calculate_lower_bound(self):
        """ Calculates the variational lower bound for current posterior.

        :return: float
            Variational lower bound.
        """
        # a_val = 2 * (self.N - 1) * np.sum(np.exp(self.log_pi_k), axis=0)
        a_val = (np.sum(np.exp(self.log_pi_k), axis=0)**2)-np.sum(np.exp(self.log_pi_k*2), axis=0)

        h_int_points_for_f = [-self.f_int_points[k]/2-self.f2_int_points[k]*self.w_int_points_f[k]/2-np.log(2) for k in range(self.K)]


        integrand_for_f = [h_int_points_for_f[k]-np.log(self.Lambda_f_without_pik[k])-logcosh(self.c_int_points_f[k]/2)  + 0.5 *(self.c_int_points_f[k]**2)*self.w_int_points_f[k]+1 for k in range(self.K)]

        h_X_for_f = self.z_ij_n_mat*(0.5*np.asarray(self.f_X).T-0.5*np.asarray(self.f2_X).T*self.w_X_f.T-np.log(2)+self.log_pi_k[self.relations[:, 0].astype(int)]+self.log_pi_k[self.relations[:, 1].astype(int)])
        summand_for_f = h_X_for_f -logcosh((0.5*self.c_X_f.T)) +0.5*((self.c_X_f**2).T)*self.w_X_f.T

        LL_f = np.sum([integrand_for_f[k].dot(self.Lambda_f_without_pik[k]*a_val[k])/self.num_integration_points*self.T for k in range(self.K)])
        LL_f += np.sum(summand_for_f)
        LL_f -= self.T*np.sum((np.sum(self.pi_k, axis=0)**2-np.sum(self.pi_k**2, axis=0)))


        Sigma_s_mugmug_f = [self.Sigma_f_s[k] + np.outer(self.f_s[k], self.f_s[k]) for k in range(self.K)]

        L_f = np.sum([-.5*np.trace(self.Ks_inv[k].dot(Sigma_s_mugmug_f[k])) for k in range(self.K)])

        L_f -= np.sum([.5*self.logdet_Ks[k] for k in range(self.K)])

        L_f += np.sum([.5*self.logdet_Sigma_f_s[k] + .5*self.num_inducing_points for k in range(self.K)])


        L_m_pi = np.sum((self.alpha_pi_0*np.log(self.beta_pi_0))[:, np.newaxis] - gammaln(self.alpha_pi_0[:, np.newaxis]) + (self.alpha_pi_0[:, np.newaxis] - 1)*self.log_pi_k - self.beta_pi_0[:, np.newaxis]*self.pi_k)
        L_m_pi += np.sum(self.alpha_pi_q - np.log(self.beta_pi_q) + gammaln(self.alpha_pi_q) + (1. - self.alpha_pi_q)*digamma(self.alpha_pi_q))

        # print('L-f: ', L_f)
        # print('L-g: ', L_g)
        # print('L-ff: ', LL_f)
        # print('L-gg: ', LL_g)
        # print('L-m-pi: ', L_m_pi)
        # print('L-z-b: ', LL_z_b)

        elbo = L_f + L_m_pi + LL_f

        return elbo

    def calculate_hyperparam_derivative_f(self):
        """ Calculates the derivative of the hyperparameters.

        :return: np.ndarray [D + 1]
            Derivatives of hyperparameters.
        """


        dL_dtheta = [np.empty(2) for _ in range(self.K)]
        a_val = (np.sum(np.exp(self.log_pi_k), axis=0)**2)-np.sum(np.exp(self.log_pi_k*2), axis=0)

        for kk in range(self.K):
            theta1 = self.cov_params[kk][0]
            theta2 = self.cov_params[kk][1]


            Sigma_s_mugmug = self.Sigma_f_s[kk] + np.outer(self.f_s[kk], self.f_s[kk])

            dks_X = np.empty([self.ks_X[0].shape[0], self.ks_X[0].shape[1],2])
            dks_int_points = np.empty([self.ks_int_points[0].shape[0], self.ks_int_points[0].shape[1], 2])
            dKs = np.empty([self.Ks[0].shape[0], self.Ks[0].shape[1], 2])

            dKss = np.zeros(2)
            dKss[0] = 1.

            # kernel derivatives wrt theta1
            dks_X[:, :, 0] = self.ks_X[kk] / theta1
            dks_int_points[:, :, 0] = self.ks_int_points[kk] / theta1
            dKs[:, :, 0] = self.Ks[kk] / theta1

            # kernel derivatives wrt theta2
            dx = self.induced_points[:, np.newaxis]-self.relations[:, 2][np.newaxis, :]
            dks_X[:, :, 1] = self.ks_X[kk][:, :] * (dx ** 2) / (theta2** 3)
            dx = self.induced_points[:, np.newaxis]-self.integration_points[np.newaxis, :]
            dks_int_points[:, :, 1] = self.ks_int_points[kk][:, :] * (dx ** 2) / (theta2** 3)
            dx = self.induced_points[:, np.newaxis]-self.induced_points[np.newaxis, :]
            dKs[:, :, 1] = self.Ks[kk] * (dx ** 2) / (theta2 ** 3)

            for itheta in range(2):
                dKs_inv = -self.Ks_inv[kk].dot(dKs[:, :, itheta].dot(self.Ks_inv[kk]))

                dkappa_X = self.Ks_inv[kk].dot(dks_X[:, :, itheta]) + dKs_inv.dot(self.ks_X[kk])
                dkappa_int_points = self.Ks_inv[kk].dot(dks_int_points[:, :, itheta]) + dKs_inv.dot(
                    self.ks_int_points[kk])

                dKtilde_X = dKss[itheta] - np.sum(dks_X[:, :, itheta] * self.kappa_X[kk], axis=0) - np.sum(
                    self.ks_X[kk] * dkappa_X, axis=0)
                dKtilde_int_points = dKss[itheta] - np.sum(dks_int_points[:, :, itheta] * self.kappa_int_points[kk],
                    axis=0) - np.sum(self.ks_int_points[kk] * dkappa_int_points,axis=0)

                dg1_X = self.f_s[kk].dot(dkappa_X)
                dg1_int_points = self.f_s[kk].dot(dkappa_int_points)

                dg2_X = (dKtilde_X + 2. * np.sum(self.kappa_X[kk]* Sigma_s_mugmug.dot(dkappa_X),axis=0)) * self.w_X_f[kk]
                dg2_X = dg2_X
                dg2_int_points = (dKtilde_int_points + 2. * np.sum(self.kappa_int_points[kk] * Sigma_s_mugmug.dot(dkappa_int_points),axis=0)) * self.w_int_points_f[kk]

                # a_val = np.sum(np.exp(self.log_pi_k[np.newaxis] + self.log_pi_k[:, np.newaxis]),
                #                axis=(0, 1)) - np.sum(np.exp(2 * self.log_pi_k), axis=(0))

                dL_dtheta[kk][itheta] = .5 * (np.sum(dg1_X*self.z_ij_n_mat[:, kk]) - np.sum(dg2_X*self.z_ij_n_mat[:, kk]))
                dL_dtheta[kk][itheta] += .5 * np.dot(-dg1_int_points - dg2_int_points,self.Lambda_f_without_pik[kk]*a_val[kk]) / self.num_integration_points * self.T
                dL_dtheta[kk][itheta] -= .5 * np.trace(self.Ks_inv[kk].dot(dKs[:, :, itheta]))
                dL_dtheta[kk][itheta] += .5 * np.trace(self.Ks_inv[kk].dot(dKs[:, :, itheta].dot(self.Ks_inv[kk].dot(Sigma_s_mugmug))))

        return dL_dtheta







    def update_hyperparameters(self, ite_val):
        """ Updates the hyperparameters with Adam.
        """
        if self.update_hyper_f:
            dL_dtheta_f = self.calculate_hyperparam_derivative_f()

            for kk in range(self.K):
                logtheta1 = np.log(self.cov_params[kk][0])
                logtheta2 = np.log(self.cov_params[kk][1])

                dL_dlogtheta1 = dL_dtheta_f[kk][0] * np.exp(logtheta1)
                dL_dlogtheta2 = dL_dtheta_f[kk][1] * np.exp(logtheta2)

                self.m_hyper_adam[kk][0] = self.beta1_adam * self.m_hyper_adam[kk][0] + (1. - self.beta1_adam) * dL_dlogtheta1
                self.v_hyper_adam[kk][0] = self.beta2_adam * self.v_hyper_adam[kk][0] + (1. - self.beta2_adam) * dL_dlogtheta1 ** 2
                self.m_hyper_adam[kk][1] = self.beta1_adam * self.m_hyper_adam[kk][1] + (1. - self.beta1_adam) * dL_dlogtheta2
                self.v_hyper_adam[kk][1] = self.beta2_adam * self.v_hyper_adam[kk][1] + (1. - self.beta2_adam) * dL_dlogtheta2 ** 2
                m_hat = self.m_hyper_adam[kk] / (1. - self.beta1_adam)
                v_hat = self.v_hyper_adam[kk] / (1. - self.beta2_adam)
                # logtheta1 += self.alpha_adam/(np.sqrt(ite_val)) * m_hat[0] / (np.sqrt(v_hat[0]) +
                #                                         self.epsilon_adam)
                # logtheta2 += self.alpha_adam/(np.sqrt(ite_val)) * m_hat[1] / (np.sqrt(v_hat[1]) +
                #                                          self.epsilon_adam)
                logtheta1 += self.alpha_adam * m_hat[0] / (np.sqrt(v_hat[0]) +self.epsilon_adam)
                logtheta2 += self.alpha_adam * m_hat[1] / (np.sqrt(v_hat[1]) +self.epsilon_adam)
                self.cov_params[kk][0] = np.exp(logtheta1)
                self.cov_params[kk][1] = np.exp(logtheta2)


        self.update_kernels()

    def update_kernels(self):
        """ Updates all kernels (for inducing, observed and integration points).
        """
        if self.update_hyper_f:
            self.Ks = [cov_func(self.induced_points, self.induced_points, self.cov_params[k]) for k in range(self.K)]
            self.ks_X = [cov_func(self.induced_points, self.relations[:, 2], self.cov_params[k]) for k in range(self.K)]
            self.ks_int_points = [cov_func(self.induced_points,self.integration_points, self.cov_params[k]) for k in range(self.K)]
            L = [np.linalg.cholesky(self.Ks[k] + self.noise*np.eye(self.Ks[k].shape[0])) for k in range(self.K)]
            L_inv = [solve_triangular(L[k], np.eye(L[k].shape[0]), lower = True, check_finite=False) for k in range(self.K)]
            self.Ks_inv = [L_inv[k].T.dot(L_inv[k]) for k in range(self.K)]
            self.logdet_Ks = [2. * np.sum(np.log(L[k].diagonal())) for k in range(self.K)]
            self.kappa_X = [self.Ks_inv[k].dot(self.ks_X[k]) for k in range(self.K)]
            self.kappa_int_points = [self.Ks_inv[k].dot(self.ks_int_points[k]) for k in range(self.K)]



    def predictive_intensity_function(self, X_eval):
        """ Computes the predictive intensity function at X_eval by Gaussian
        quadrature.

        :param X_eval: numpy.ndarray [num_points_eval x D]
            Points where the intensity function should be evaluated.

        :returns:
            numpy.ndarray [num_points]: mean of predictive posterior intensity
            numpy.ndarray [num_points]: variance of predictive posterior
                                        intensity
        """
        num_preds = len(X_eval)
        mu_pred, var_pred = self.predictive_posterior_GP_f(X_eval)

        mean_lmbda_pred, var_lmbda_pred = np.empty(num_preds), \
                                          np.empty(num_preds)

        # mean_lmbda_q1 = self.lmbda_star_q1
        # var_lmbda_q1 = self.alpha_q1/(self.beta_q1**2)
        # mean_lmbda_q1_squared = var_lmbda_q1 + mean_lmbda_q1**2

        for ipred in range(num_preds):
            for k in range(self.K):
                for ii in range(self.N):
                    for jj in range(self.N):
                        mu, std = mu_pred[k][ipred], np.sqrt(var_pred[k][ipred])
                        func1 = lambda g_pred: 1. / (1. + np.exp(-g_pred)) * \
                                              np.exp(-.5*(g_pred - mu)**2 / std**2) / \
                                              np.sqrt(2.*np.pi*std**2)
                        a, b = mu - 10.*std, mu + 10.*std
                        mean_lmbda_pred[ipred, k, ii, jj] = self.pi_k[ii,k]*self.pi_k[jj, k]*quadrature(func1, a, b,
                                                                          maxiter=500)[0]
                        func2 = lambda g_pred: (1. / (1. + np.exp(-g_pred)))**2 * \
                                              np.exp(-.5*(g_pred - mu)**2 / std**2) / \
                                              np.sqrt(2.* np.pi*std**2)
                        a, b = mu - 10. * std, mu + 10. * std
                        mean_lmbda_pred_squared = mean_lmbda_q1_squared *\
                                                 quadrature(func2, a, b, maxiter=500)[0]
                        var_lmbda_pred[ipred] = mean_lmbda_pred_squared - mean_lmbda_pred[
                                                                        ipred]**2

        return mean_lmbda_pred, var_lmbda_pred
