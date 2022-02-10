# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3, profile=False
import numpy as np
import time
from scipy.special import gammaln, logsumexp
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, exp, sqrt, tanh, cosh

def PG_g_elbo(double[:, :] z_ij_n_mat, int num_integration_points, int K, int NumE, double[:,:] g_int_points, double[:,:] g2_int_points, double[:] log_m, \
         double[:] eventtimes, double[:] integration_points, double T):
    cdef:
        int k, int_i, E_i
        double[:, :] integration_LogL_L = np.zeros((NumE, K))
        double[:, :] integration_z_g2_w_L = np.zeros((NumE, K))
        double[:, :] integration_z_g_L = np.zeros((NumE, K))
        double[:, :] integration_log_cosh_c_L = np.zeros((NumE, K))
        double[:, :] integration_c2_w_L = np.zeros((NumE, K))
        double[:, :] integration_L = np.zeros((NumE, K))
        double c_int_points_g_k, mu_w_int_points_g_k, Lambda_g_k, w_L_val_temp

    for int_i in range(num_integration_points):
        for E_i in range(NumE):
            if integration_points[int_i] < (T-eventtimes[E_i]):
                for k in range(K):
                    c_int_points_g_k = sqrt(z_ij_n_mat[E_i, k]*g2_int_points[int_i, k])
                    mu_w_int_points_g_k = 0.5*tanh(0.5*c_int_points_g_k)/c_int_points_g_k
                    Lambda_g_k = exp(log_m[k]-0.5*z_ij_n_mat[E_i, k]*g_int_points[int_i, k])/\
                                     cosh(0.5*(c_int_points_g_k))*(2**(-z_ij_n_mat[E_i, k]))
                    w_L_val_temp = mu_w_int_points_g_k*Lambda_g_k
                    
                    integration_L[E_i, k] += (Lambda_g_k)
                    integration_z_g2_w_L[E_i, k] += (z_ij_n_mat[E_i, k]*g2_int_points[int_i, k]*w_L_val_temp)
                    integration_z_g_L[E_i, k] += (z_ij_n_mat[E_i, k]*g_int_points[int_i, k]*Lambda_g_k)
                    integration_LogL_L[E_i, k] += (log(Lambda_g_k)*Lambda_g_k)
                    integration_log_cosh_c_L[E_i, k] += (log(cosh(c_int_points_g_k/2))*Lambda_g_k)
                    integration_c2_w_L[E_i, k] += ((c_int_points_g_k**2)/2*mu_w_int_points_g_k*Lambda_g_k)

    return integration_z_g2_w_L, integration_z_g_L, integration_LogL_L, integration_log_cosh_c_L, integration_c2_w_L, integration_L


def PG_g_z(double[:, :] z_ij_n_mat, int num_integration_points, int K, int NumE, double[:,:] g_int_points, double[:,:] g2_int_points, double[:] log_m, \
         double[:] eventtimes, double[:] integration_points, double T):
    cdef:
        int k, int_i, E_i
        double[:, :] integration_L = np.zeros((NumE, K))
        double[:, :] integration_g2_w_L = np.zeros((NumE, K))
        double[:, :] integration_g_L = np.zeros((NumE, K))
        double c_int_points_g_k, mu_w_int_points_g_k, Lambda_g_k, w_L_val_temp

    for int_i in range(num_integration_points):
        for E_i in range(NumE):
            if integration_points[int_i] < (T-eventtimes[E_i]):
                for k in range(K):
                    c_int_points_g_k = sqrt(z_ij_n_mat[E_i, k]*g2_int_points[int_i, k])
                    mu_w_int_points_g_k = 0.5*tanh(0.5*c_int_points_g_k)/c_int_points_g_k
                    Lambda_g_k = exp(log_m[k]-0.5*z_ij_n_mat[E_i, k]*g_int_points[int_i, k])/\
                                     cosh(0.5*(c_int_points_g_k))*(2**(-z_ij_n_mat[E_i, k]))
                    w_L_val_temp = mu_w_int_points_g_k*Lambda_g_k

                    integration_L[E_i, k] += (Lambda_g_k)
                    integration_g2_w_L[E_i, k] += (g2_int_points[int_i, k]*w_L_val_temp)
                    integration_g_L[E_i, k] += (g_int_points[int_i, k]*Lambda_g_k)

    return integration_g2_w_L, integration_g_L, integration_L


def PG_g_m(double[:, :] z_ij_n_mat, int num_integration_points, int K, int NumE, double[:,:] g_int_points, double[:,:] g2_int_points, double[:] log_m, \
         double[:] eventtimes, double[:] integration_points, double T):
    cdef:
        int k, int_i, E_i
        double[:, :] integration_L = np.zeros((NumE, K))
        double c_int_points_g_k, mu_w_int_points_g_k, Lambda_g_k, w_L_val_temp

    for int_i in range(num_integration_points):
        for E_i in range(NumE):
            if integration_points[int_i] < (T-eventtimes[E_i]):
                for k in range(K):
                    c_int_points_g_k = sqrt(z_ij_n_mat[E_i, k]*g2_int_points[int_i, k])
                    mu_w_int_points_g_k = 0.5*tanh(0.5*c_int_points_g_k)/c_int_points_g_k
                    Lambda_g_k = exp(log_m[k]-0.5*z_ij_n_mat[E_i, k]*g_int_points[int_i, k])/\
                                     cosh(0.5*(c_int_points_g_k))*(2**(-z_ij_n_mat[E_i, k]))

                    integration_L[E_i, k] += (Lambda_g_k)

    return integration_L


def PG_g_sum(double[:, :] z_ij_n_mat, int num_integration_points, int K, int NumE, double[:,:] g_int_points, double[:,:] g2_int_points, double[:] log_m, \
         double[:] eventtimes, double[:] integration_points, double T):
    cdef:
        int k, int_i, E_i
        double[:, :] sum_z_L = np.zeros((num_integration_points, K))
        double[:, :] sum_z_w_L = np.zeros((num_integration_points, K))
        double c_int_points_g_k, mu_w_int_points_g_k, Lambda_g_k, w_L_val_temp

    for int_i in range(num_integration_points):
        for E_i in range(NumE):
            if integration_points[int_i] < (T-eventtimes[E_i]):
                for k in range(K):
                    c_int_points_g_k = sqrt(z_ij_n_mat[E_i, k]*g2_int_points[int_i, k])
                    mu_w_int_points_g_k = 0.5*tanh(0.5*c_int_points_g_k)/c_int_points_g_k
                    Lambda_g_k = exp(log_m[k]-0.5*z_ij_n_mat[E_i, k]*g_int_points[int_i, k])/cosh(0.5*(c_int_points_g_k))*(2**(-z_ij_n_mat[E_i, k]))
                    w_L_val_temp = mu_w_int_points_g_k*Lambda_g_k

                    sum_z_L[int_i, k] += (z_ij_n_mat[E_i, k]*Lambda_g_k)
                    sum_z_w_L[int_i, k] += (z_ij_n_mat[E_i, k]*w_L_val_temp)

    return sum_z_L, sum_z_w_L

def PG_g_sum_endogenous_rate(int num_integration_points, int K, int NumE, double[:,:] Lambda_g_v, double[:,:] a1, double[:, :] w_int_points_g, double[:] eventtimes, double[:] integration_points, double T):
    cdef:
        int kk, int_i, E_i
        double[:, :] inte_L = np.zeros((num_integration_points, K))
        double[:, :] inte_w_L = np.zeros((num_integration_points, K))
        double Lambda_int_e, w_L_val_temp

    for int_i in range(num_integration_points):
        for E_i in range(NumE):
            if integration_points[int_i] < (T-eventtimes[E_i]):
                for kk in range(K):
                    Lambda_int_e = Lambda_g_v[int_i,kk]*a1[E_i,kk]
                    w_L_val_temp = w_int_points_g[int_i, kk]*Lambda_int_e

                    inte_L[int_i, kk] += (Lambda_int_e)
                    inte_w_L[int_i, kk] += (w_L_val_temp)

    return inte_L, inte_w_L


def gradient_g_part(double[:] z_ij_n_mat_k, int num_integration_points, int NumE, double[:] g_int_points_k, double[:] g2_int_points_k, double log_m_k, \
         double[:] eventtimes, double[:] integration_points, double T, double[:] dg_1_int_points, double[:] dg_2_int_points):
    cdef:
        int k, int_i, E_i
        double[:] gradient_g_part_val= np.zeros(NumE)
        double c_int_points_g_k, mu_w_int_points_g_k, Lambda_g_k, w_L_val_temp

    for int_i in range(num_integration_points):
        for E_i in range(NumE):
            if integration_points[int_i] < (T-eventtimes[E_i]):
                c_int_points_g_k = sqrt(z_ij_n_mat_k[E_i]*g2_int_points_k[int_i])
                mu_w_int_points_g_k = 0.5*tanh(0.5*c_int_points_g_k)/c_int_points_g_k
                Lambda_g_k = exp(log_m_k-0.5*z_ij_n_mat_k[E_i]*g_int_points_k[int_i])/\
                                 cosh(0.5*(c_int_points_g_k))*(2**(-z_ij_n_mat_k[E_i]))
                gradient_g_part_val[E_i] += Lambda_g_k*z_ij_n_mat_k[E_i]*(dg_1_int_points[int_i]+dg_2_int_points[int_i]*mu_w_int_points_g_k)

    return gradient_g_part_val


def gradient_g_part_K_endogenous_rate(int num_integration_points, int NumE, double[:] Lambda_g_v_k, double[:] a1_k, double[:] w_int_points_g_k, double[:] eventtimes, double[:] integration_points, double T, double[:] dg_1_int_points, double[:] dg_2_int_points):
    cdef:
        int k, int_i, E_i
        double[:] gradient_g_part_val= np.zeros(NumE)
        double Lambda_g_m_int_i

    for int_i in range(num_integration_points):
        for E_i in range(NumE):
            if integration_points[int_i] < (T-eventtimes[E_i]):
                Lambda_g_m_int_i = Lambda_g_v_k[int_i]*a1_k[E_i]
                gradient_g_part_val[E_i] += Lambda_g_m_int_i*(dg_1_int_points[int_i]+dg_2_int_points[int_i]*w_int_points_g_k[int_i])

    return gradient_g_part_val


def cal_LL_g_K_endogenous_rate(int num_integration_points, int NumE, double[:, :] Lambda_g_v, double[:, :] a1, double[:, :] inte_val, double[:] eventtimes, double[:] integration_points, double T, int K):
    cdef:
        int k, int_i, E_i
        double[:, :] ll_g_inte = np.zeros((NumE, K))

    for int_i in range(num_integration_points):
        for E_i in range(NumE):
            if integration_points[int_i] < (T-eventtimes[E_i]):
                for k in range(K):
                    ll_g_inte[E_i, k] += Lambda_g_v[int_i, k]*a1[E_i, k]*inte_val[k, int_i]

    return ll_g_inte


def cal_ll_integration_g(double[:] eval_points, double[:] time_period, double[:, :] z_ij_n_mat, double[:] eventtime, double[:] m_k, int[:] index_m, double[:,:] sigma_g_general, int K):

    cdef:
        double integration_g = 0., influence_range, val_i
        int index_i, index_num, pi_index, kk

    for index_i in index_m:
        influence_range = (time_period[1]-eventtime[index_i])
        val_i = 0.
        index_num = 0
        for pi_index in range(len(eval_points)):
            if eval_points[pi_index] <influence_range:
                for kk in range(K):
                    val_i += m_k[kk]*sigma_g_general[pi_index,kk]*z_ij_n_mat[index_i,kk]
                index_num += 1
            else:
                break
        integration_g += val_i/index_num * influence_range

    return -integration_g


def cal_ll_integration_g_endogenous_function(double[:] eval_points, double[:] time_period, double[:] eventtime, double[:,:] m_k, int[:] index_m, int[:, :] nodal_pair, double[:,:] sigma_g_general, int K):

    cdef:
        double integration_g = 0., influence_range, val_i
        int index_i, index_num, pi_index, kk

    for index_i in index_m:
        influence_range = (time_period[1]-eventtime[index_i])
        val_i = 0.
        index_num = 0
        for pi_index in range(len(eval_points)):
            if eval_points[pi_index] <influence_range:
                for kk in range(K):
                    val_i += m_k[nodal_pair[index_i,0], kk]*m_k[nodal_pair[index_i,1], kk]*sigma_g_general[pi_index,kk]
                index_num += 1
            else:
                break
        integration_g += val_i/index_num * influence_range

    return -integration_g


def cal_ll_integration_g_future(double[:] eval_points, int[:, :] nodepair, double[:] time_period, double[:, :] z_ij_n_mat, double[:] eventtime, double[:] m_k, double[:,:] sigma_g_general, int K ,int NumE, int N):

    cdef:
        double maximum_range, minimum_range, val_i
        int index_i, index_num, pi_index, kk
        double[:, :] integration_g = np.zeros((N, N))

    for index_i in range(NumE):
        maximum_range = (time_period[1]-eventtime[0])
        minimum_range = (time_period[0]-eventtime[0])
        val_i = 0.
        index_num = 0
        for pi_index in range(len(eval_points)):
            if eval_points[pi_index] <maximum_range:
                if eval_points[pi_index] > minimum_range:
                    for kk in range(K):
                        val_i += m_k[kk]*sigma_g_general[pi_index,kk]*z_ij_n_mat[index_i,kk]
                    index_num += 1
            else:
                break
        integration_g[nodepair[index_i, 1]][nodepair[index_i, 0]] += val_i/index_num * (maximum_range-minimum_range)

    return integration_g


def cal_ll_integration_g_future_endogenous_function(double[:] eval_points, int[:, :] nodepair, double[:] time_period, double[:] eventtime, double[:,:] m_k, double[:,:] sigma_g_general, int K ,int NumE, int N):

    cdef:
        double maximum_range, minimum_range, val_i
        int index_i, index_num, pi_index, kk
        double[:, :] integration_g = np.zeros((N, N))

    for index_i in range(NumE):
        maximum_range = (time_period[1]-eventtime[0])
        minimum_range = (time_period[0]-eventtime[0])
        val_i = 0.
        index_num = 0
        for pi_index in range(len(eval_points)):
            if eval_points[pi_index] <maximum_range:
                if eval_points[pi_index] > minimum_range:
                    for kk in range(K):
                        val_i += m_k[nodepair[index_i, 0], kk]*m_k[nodepair[index_i, 1], kk]*sigma_g_general[pi_index,kk]
                    index_num += 1
            else:
                break
        integration_g[nodepair[index_i, 1]][nodepair[index_i, 0]] += val_i/index_num * (maximum_range-minimum_range)

    return integration_g


def cal_ll_integration_g_no_clustering(double[:] eval_points, double[:] time_period, double[:] eventtime, double[:] m_k, int[:] index_m, double[:,:] sigma_g_general, int K):

    cdef:
        double integration_g = 0., influence_range, val_i
        int index_i, index_num

    for index_i in index_m:
        influence_range = (time_period[1]-eventtime[index_i])
        val_i = 0.
        index_num = 0
        for pi_index in range(len(eval_points)):
            if eval_points[pi_index] <influence_range:
                for kk in range(K):
                    val_i += m_k[kk]*sigma_g_general[pi_index,kk]
                index_num += 1
            else:
                break
        integration_g += val_i/index_num * influence_range

    return -integration_g


def judge_integration_relation(double[:] eventtimes, double[:] integration_points, int NumE, int num_integration_points, double T):
    cdef:
        int int_i, E_i
        int[:] judge_per_inte_points = np.zeros(num_integration_points, dtype = np.intc)
        int[:] judge_per_relation = np.zeros(NumE, dtype = np.intc)

    for int_i in range(num_integration_points):
        for E_i in range(NumE):
            if integration_points[int_i] < (T-eventtimes[E_i]):
                judge_per_inte_points[int_i] += 1
                judge_per_relation[E_i] += 1

    return judge_per_relation, judge_per_inte_points


def z_part_2(double[:] b_non_0_1D, double[:, :] second_part, int[:] ji_n_slash_index, int[:] diff_after_len, int NumE, int KK):
    cdef:
        int k1, E_i
        int non_0_index = 0
        double[:, :] part_2 = np.zeros((NumE, KK))

    for E_i in range(NumE):
        if diff_after_len[E_i]>0:
            for _ in range(diff_after_len[E_i]):
                for k1 in range(KK):
                    part_2[E_i, k1] += b_non_0_1D[ji_n_slash_index[non_0_index]]*second_part[ji_n_slash_index[non_0_index], k1]
                non_0_index += 1

    return part_2


def z_part_2_K_endogenous_rate(double[:, :] b_non_0_1D, int[:] diff_before_len, int NumE, int KK):
    cdef:
        int kk, E_i
        int non_0_index = 0
        double[:, :] part_2 = np.zeros((NumE, KK))

    for E_i in range(NumE):
        if diff_before_len[E_i]>0:
            for _ in range(diff_before_len[E_i]):
                for kk in range(KK):
                    part_2[E_i, kk] += b_non_0_1D[non_0_index, kk]
                non_0_index += 1

    return part_2


def cal_g_integration_m(Lambda_g_without_v, a1, K, eventtime_i, integration_points, T):
    cdef:
        double[:, :] integration_m = np.zeros((len(eventtime_i), K))
        int kk, E_m, int_i
    for int_i in range(len(integration_points)):
        for E_m in range(len(eventtime_i)):
            if integration_points[int_i] < (T-eventtime_i[E_m]):
                for kk in range(K):
                    integration_m[E_m, kk] += a1[E_m,kk]*(Lambda_g_without_v[int_i, kk])

    return integration_m


def z_part_2_no_clustering(double[:] b_non_0_1D, double[:, :] second_part, int[:] diff_before_len, int NumE, int KK):
    cdef:
        int k1, E_i
        int non_0_index = 0
        double[:, :] part_2 = np.zeros((NumE, KK))

    for E_i in range(NumE):
        if diff_before_len[E_i]>0:
            for _ in range(diff_before_len[E_i]):
                for k1 in range(KK):
                    part_2[E_i, k1] += b_non_0_1D[non_0_index]*second_part[non_0_index, k1]
                non_0_index += 1

    return part_2


def process_b_ij_n(double[:] b_ij_n_0, double[:] b_ij_n_nslash, int[:] diff_before_len, int NumE, int before_max):
    cdef:
        int E_i, before_i
        int before_index = 0, current_step=0
        double[:] log_val = np.zeros(before_max)
        double[:] b_0_1D = np.ones(NumE)
        double[:] b_non_0_1D = np.ones(len(b_ij_n_nslash))
        double max_log, pp_sum, logsumexp_val
    for E_i in range(NumE):
        if diff_before_len[E_i]>0:
            log_val[0] = b_ij_n_0[E_i]
            max_log = log_val[0]
            for before_i in range(1, diff_before_len[E_i]+1):
                log_val[before_i] = b_ij_n_nslash[before_index]
                max_log = max(max_log, log_val[before_i])
                before_index += 1
            pp_sum = 0
            for before_i in range(diff_before_len[E_i]+1):
                log_val[before_i] -= max_log
                pp_sum += exp(log_val[before_i])
            logsumexp_val = log(pp_sum)
            for before_i in range(diff_before_len[E_i]+1):
                log_val[before_i] -= logsumexp_val
                if before_i == 0:
                    b_0_1D[E_i] = exp(log_val[before_i])+1e-16
                else:
                    b_non_0_1D[current_step+before_i-1] = exp(log_val[before_i])+1e-16
            current_step = before_index

    return b_0_1D, b_non_0_1D

def process_z_ij_n_endogenous_rate(double[:, :] first_part, double[:, :] second_part, int[:] diff_before_len, int NumE, int before_max, int K):
    cdef:
        int E_i, before_i, kk
        int before_index = 0, current_step=0
        double[:, :] log_val = np.zeros((before_max, K))
        double[:, :] b_0_1D = np.ones((NumE, K))
        double[:, :] b_non_0_1D = np.ones((len(second_part), K))
        double max_log, pp_sum, logsumexp_val

    for E_i in range(NumE):

        max_log = first_part[E_i,0]
        for kk in range(K):
            log_val[0,kk] = first_part[E_i, kk]
            max_log = max(max_log, log_val[0,kk])

        if diff_before_len[E_i]==0:
            pp_sum = 0
            for kk in range(K):
                log_val[0,kk] -= max_log
                pp_sum += exp(log_val[0][kk])
            logsumexp_val = log(pp_sum)
            for kk in range(K):
                log_val[0,kk] -= logsumexp_val
                b_0_1D[E_i][kk] = exp(log_val[0,kk])+1e-16

        elif diff_before_len[E_i]>0:

            for before_i in range(1, diff_before_len[E_i]+1):
                for kk in range(K):
                    log_val[before_i][kk] = second_part[before_index][kk]
                    max_log = max(max_log, log_val[before_i][kk])
                before_index += 1
            pp_sum = 0.
            for before_i in range(diff_before_len[E_i]+1):
                for kk in range(K):
                    log_val[before_i][kk] -= max_log
                    pp_sum += exp(log_val[before_i][kk])

            logsumexp_val = log(pp_sum)

            for before_i in range(diff_before_len[E_i]+1):
                for kk in range(K):
                    log_val[before_i][kk] -= logsumexp_val
                    if before_i == 0:
                        b_0_1D[E_i][kk] = exp(log_val[before_i][kk])+1e-16
                    else:
                        b_non_0_1D[current_step+before_i-1][kk] = exp(log_val[before_i][kk])+1e-16
            current_step = before_index

    return b_0_1D, b_non_0_1D
