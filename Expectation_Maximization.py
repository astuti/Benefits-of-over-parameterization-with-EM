import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style("white")
%matplotlib inline
import numpy as np #for generating data
from math import sqrt, log, exp, pi
import random
import itertools
import csv


'''
Generate data-set
'''
# ******
separation = 2
means = [4., 6.]
weights = [.9, .1]

print("separation: ", separation)
print("means: ", means)
stdevs = [2., 5.]
data_pts_per_gaussian = 1000
# ******

data_pts = 1000
data = np.array([])
g1 = 0
g2 = 0

for i in range(0, data_pts):
    r = random.random()
    if (r < weights[0]):
        data = np.append(data, np.random.normal(means[0], stdevs[0], 1))
        g1 += 1.
    else:
        data = np.append(data, np.random.normal(means[1], stdevs[1], 1))
        g2 += 1.

print(g1 / (g1 + g2))

# For data visiualisation calculate left and right of the graph
min_graph = min(data)
max_graph = max(data)
x = np.linspace(min_graph, max_graph, data_pts)
sb.distplot(data, bins=20, kde=False)


class Gaussian:
    "basic details of a gaussian"

    def __init__(self, mu, sigma, weight):
        self.mu = mu
        self.sigma = sigma
        self.weight = weight

        # storing the updates of EM iterations
        self.iterating_mu = [[] for x in range(0, 2)]
        self.iterating_sigma = [[] for x in range(0, 2)]
        self.iterating_weight = [[] for x in range(0, 2)]

    # probability density function
    def pdf(self, datum):
        "Probability of a data point given the current parameters"
        u = (datum - self.mu) / abs(self.sigma)
        y = (1 / (sqrt(2 * pi) * abs(self.sigma))) * exp(-u * u / 2)
        return y


class GaussianMixture:
    def __init__(self, num_gauss, num_data, data, mu_list, sigma_list, weight_list):
        self.num_gaussian = num_gauss
        self.num_data_pts = num_data
        self.data = data
        self.gamma_matrix = [[0 for x in range(self.num_gaussian + 1)] for y in range(self.num_data_pts)]
        self.gaussian_list = [0 for x in range(self.num_gaussian)]

        for i in range(0, self.num_gaussian):
            self.gaussian_list[i] = Gaussian(mu_list[i], sigma_list[i], weight_list[i])

        # calculate loglikelihood here
        self.loglike = -1.

    def e_step(self):
        "calculate responsibilities"
        for dp in range(0, self.num_data_pts):
            # for each data point
            sum_responsibilities_point = 0.
            for cls in range(0, self.num_gaussian):
                # storing numerator of responsibility
                self.gamma_matrix[dp][cls] = self.gaussian_list[cls].weight * self.gaussian_list[cls].pdf(self.data[dp])
                # to store sum of numerator for each data point(and all guasssians)
                sum_responsibilities_point += self.gamma_matrix[dp][cls]
            self.gamma_matrix[dp][self.num_gaussian] = sum_responsibilities_point

    def m_step(self, fixed_mean, fixed_sigma, fixed_weights):
        "Differentiate and calculate mu, sigma, weights"
        for gauss in range(0, self.num_gaussian):
            # calculate mu, variance, weights, Nk
            mu_Nk = 0.
            var_Nk = 0.
            Nk = 0.

            # 1. Calculate mean and store iteration values
            for dp in range(0, self.num_data_pts):
                mu_Nk += (self.gamma_matrix[dp][gauss] / self.gamma_matrix[dp][self.num_gaussian]) * self.data[dp]
                Nk += self.gamma_matrix[dp][gauss] / self.gamma_matrix[dp][self.num_gaussian]

            mu = mu_Nk / Nk;
            self.gaussian_list[gauss].iterating_mu[0] += [self.gaussian_list[gauss].mu]
            self.gaussian_list[gauss].iterating_mu[1] += [mu]

            # 2. Calculate sigma and store iteration values
            if (fixed_sigma == 0):
                for dp in range(0, self.num_data_pts):
                    var_Nk += (self.gamma_matrix[dp][gauss] / self.gamma_matrix[dp][self.num_gaussian]) \
                              * (self.data[dp] - mu) * (self.data[dp] - mu)

                sigma = sqrt(var_Nk / Nk)
                self.gaussian_list[gauss].iterating_sigma[0] += [self.gaussian_list[gauss].sigma]
                self.gaussian_list[gauss].iterating_sigma[1] += [sigma]

            # 3. Calculate weight and store iteration values
            if (fixed_weights == 0):
                weight = Nk / self.num_data_pts
                self.gaussian_list[gauss].iterating_weight[0] += [self.gaussian_list[gauss].weight]
                self.gaussian_list[gauss].iterating_weight[1] += [weight]

                # Update mean, sigma and weight for the gaussian
                self.gaussian_list[gauss].mu = mu
                self.gaussian_list[gauss].sigma = sigma
                self.gaussian_list[gauss].weight = weight

                # Compute likelihood
        sum_outerloop_log = 0.
        for dp in range(0, self.num_data_pts):
            sum_innerloop = 0.
            for g in range(0, self.num_gaussian):
                sum_innerloop += self.gaussian_list[g].weight * self.gaussian_list[g].pdf(self.data[dp])
            sum_outerloop_log += np.log([sum_innerloop])[0]
        self.loglike = sum_outerloop_log

    def pdf_mixture(self, d):
        pdf = 0.;
        for gaussian in self.gaussian_list:
            pdf += gaussian.weight * gaussian.pdf(d)
        return pdf

    def em_iteration(self):
        "Iterate, compute log likelihood and check convergence"
        self.e_step()
        self.m_step()


'''
Initialization
'''


def run_tests(writer):
    num_gaussian = 2

    sigma_list = [5, 15]
    weight_list = [.9, .1]

    # Number of tests to run
    num_tests = 2500
    num_success = 0
    for test in range(0, num_tests):
        mu_list = [random.choice(data), random.choice(data)]
        # print("randomly initializing mean: ", mu_list)
        mixture = GaussianMixture(num_gaussian, data_pts, data, mu_list, sigma_list, weight_list)
        optimum_likelihood = float('-inf')

        # Check convergence of log likelihood
        for j in range(0, 200):
            mixture.e_step()
            mixture.m_step(0, 0, 0)
            if (abs(mixture.loglike - optimum_likelihood) < 1e-15 or (j == 199)):
                break
            if (mixture.loglike > optimum_likelihood):
                optimum_likelihood = mixture.loglike

        if (test == 0):
            plot_graphs(mixture)

        # Check success
        error = float('+inf')
        all_calc_means = []
        all_calc_sigmas = []
        all_calc_weights = []
        for dist in range(0, mixture.num_gaussian):
            all_calc_means += [mixture.gaussian_list[dist].mu]
            all_calc_sigmas += [mixture.gaussian_list[dist].sigma]
            all_calc_weights += [mixture.gaussian_list[dist].weight]

        writer.writerow({'randomly_initialized_means': mu_list, 'calculated_means': all_calc_means, \
                         'calculated_sigmas': all_calc_sigmas, 'calculated_weights': all_calc_weights})

        # print("calculated means: " , all_calc_means)
        # print("calculated sigmas: " , all_calc_sigmas)
        # print("calculated weights: " , all_calc_weights)
        # print("====================================")
        permutations = list(itertools.permutations(all_calc_means))

        for per in permutations:
            err_list = [(a - b) ** 2 for a, b in zip(per, means)]
            weighted_err_list = [(a * b) for a, b in zip(err_list, weights)]
            permt_err = sum(weighted_err_list)
            if (permt_err < error):
                error = permt_err

        if (error < .004):
            num_success += 1

    prob_success = num_success / num_tests
    print("Probability of success:")
    print(prob_success)


with open('/Users/vulcan/paper_em/results_tests_em_paper.csv', mode='w') as csv_file:
    fieldnames = ['randomly_initialized_means', 'calculated_means', 'calculated_sigmas', \
                  'calculated_weights']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    run_tests(writer)
