#!/usr/bin/python

import sys
import math
import numpy as np
import pandas as pd
import gamdist as gam

def test_linear_regression(plot_flag=False, save_flag=False, load_flag=False):
    if load_flag:
        mdl = gam.GAM(load_from_file='test_linear_regression_model.pckl')
    else:
        mdl = gam.GAM('normal', name='test_linear_regression')
        mdl.add_feature(name='purchases', type='linear', transform=np.log1p)
        mdl.add_feature(name='gender', type='categorical')
        mdl.add_feature(name='country', type='categorical')

        X, y = generate_data(1000)
        mdl.fit(X, y, verbose=False, plot_convergence=plot_flag, save_flag=save_flag)

    mdl.summary()

    Xtest, ytest = generate_data(100)
    yhat = mdl.predict(Xtest)
    err = ytest - yhat
    print 'MSE:', err.dot(err) / len(err) # MSE

def test_logistic_regression(plot_flag=False, save_flag=False, load_flag=False):
    if load_flag:
        mdl = gam.GAM(load_from_file='test_logistic_regression_model.pckl')
    else:
        mdl = gam.GAM('binomial', name='test_logistic_regression')
        mdl.add_feature(name='purchases', type='linear', transform=np.log1p)
        mdl.add_feature(name='gender', type='categorical')
        mdl.add_feature(name='country', type='categorical')

        X, y = generate_data(1000, link=_logit_link, family=_binomial_family)
        mdl.fit(X, y, verbose=False, plot_convergence=plot_flag, save_flag=save_flag)

    mdl.summary()

    # Get the "true" probabilities, ytest
    Xtest, mu_test = generate_data(100, link=_logit_link,
                                   family=_binomial_family,
                                   return_mean=True)
    mu_hat = mdl.predict(Xtest)
    err = mu_test - mu_hat
    print 'MSE:', err.dot(err) / len(err) # MSE

def test_logistic_regression_covariate_classes(plot_flag=False, save_flag=False,
                                               load_flag=False):
    if load_flag:
        mdl = gam.GAM(load_from_file='test_logistic_regression_cc_model.pckl')
    else:
        mdl = gam.GAM('binomial', estimate_overdispersion=True,
                      name='test_logistic_regression_cc')
        mdl.add_feature(name='gender', type='categorical')
        mdl.add_feature(name='country', type='categorical')

        X, y, ccs = generate_covariate_class_data()
        mdl.fit(X, y, covariate_class_sizes=ccs,
                plot_convergence=plot_flag,
                save_flag=save_flag)

    mdl.summary()

    # Get the "true" probabilities, ytest
    Xtest, mu_test, ccs = generate_covariate_class_data(return_mean=True)
    mu_hat =  mdl.predict(Xtest)
    err = mu_test - mu_hat
    print 'MSE:', err.dot(err) / len(err) # MSE

def test_spline_regression(plot_flag=False, save_flag=False, load_flag=False):
    if load_flag:
        mdl = gam.GAM(load_from_file='test_spline_regression_model.pckl')
    else:
        mdl = gam.GAM('normal', name='test_spline_regression')
        mdl.add_feature(name='hft', type='spline', rel_dof=9.)

        X, y = generate_spline_data(1000)
        mdl.fit(X, y, verbose=False, plot_convergence=plot_flag, save_flag=save_flag)

    if plot_flag:
        mdl.plot('hft', true_fn=lambda x: np.sin(12.*(x + 0.2)) / (x + 0.2))

def test_cross_validation():
    mdl = gam.GAM('normal', name='test_additive_regression')
    mdl.add_feature(name='hft', type='spline', rel_dof=9.)

    num_training_examples = 1000
    X, y = generate_spline_data(num_training_examples)

    # Use K-fold cross validation to estimate the optimal smoothing parameter
    K = 5
    ii = np.random.permutation(num_training_examples)
    num_smooths = 20
    dev = np.zeros((num_smooths,))
    smoothing = np.linspace(0.5, 5.0, num_smooths)

    for j in range(num_smooths):
        for i in range(K):
            ia = int(i * float(num_training_examples) / K)
            ib = int((i + 1) * float(num_training_examples) / K) - 1

            traini = np.append(ii[0:ia], ii[ib:num_training_examples])
            testi = ii[ia:ib]

            Xtraini = X.iloc[traini, :]
            ytraini = y[traini]
            Xtesti = X.iloc[testi, :]
            ytesti = y[testi]

            mdl.fit(Xtraini, ytraini, smoothing=smoothing[j])
            dev[j] += mdl.deviance(Xtesti, ytesti) / np.size(ytesti)

        dev[j] /= K

    # Refit model using entire training set and best smoothing parameter
    best_smoothing = np.argmin(dev)
    mdl.fit(X, y, smoothing=smoothing[best_smoothing])
    mdl.plot('hft', true_fn=gmu_hft)

    mdl.summary()

    Xtest, ytest = generate_spline_data(100)
    yhat = mdl.predict(Xtest)
    err = ytest - yhat
    print 'MSE:', err.dot(err) / len(err)

def test_additive_regression(plot_flag=False, save_flag=False, load_flag=False):
    if load_flag:
        mdl = gam.GAM(load_from_file='test_additive_regression_model.pckl')
    else:
        mdl = gam.GAM('normal', name='test_additive_regression')
        mdl.add_feature(name='hft', type='spline', rel_dof=9.)
        mdl.add_feature(name='purchases', type='linear', transform=np.log1p)
        mdl.add_feature(name='gender', type='categorical')
        mdl.add_feature(name='country', type='categorical')

        X, y = generate_data(1000, include_hft=True)
        mdl.fit(X, y, verbose=False, plot_convergence=plot_flag, save_flag=save_flag)

    if plot_flag:
        mdl.plot('hft', true_fn=gmu_hft)

    mdl.summary()

    Xtest, ytest = generate_data(100, include_hft=True)
    yhat = mdl.predict(Xtest)
    err = ytest - yhat
    print 'MSE:', err.dot(err) / len(err) # MSE

def _identity_link(x):
    return x

def _logit_link(x):
    # This is actually the *inverse* link function
    return np.exp(x) / (1. + np.exp(x))

def _gaussian_family(mu):
    # Add noise with variance 0.1
    # For reference, the test data set tends to have a variance of 0.15,
    # so the signal to noise ratio is about 1.5.
    return mu + np.random.normal(size=mu.shape, loc=0.0, scale=np.sqrt(0.1))

def _binomial_family(mu, ccs=1):
    return np.random.binomial(ccs, p=mu)

def gmu_purchases(x):
    return 0.1*np.log1p(x) + 0.3

def gmu_gender(x):
    z = np.zeros(x.shape)
    z[x == 'male'] = 0.1
    z[x == 'female'] = -0.5
    return z

def gmu_country(x):
    z = np.zeros(x.shape)
    z[x == 'USA'] = -0.2
    z[x == 'CAN'] = 0.3
    z[x == 'GBR'] = 0.4
    return z

def gmu_hft(x):
    # sin(12 * (x + 0.2)) / (x + 0.2)
    # This is from Equation 5.22 of Hastie, Friedman, Tibshirani, "Elements of Statistical Learning"
    return np.sin(12.*(x + 0.2)) / (x + 0.2)

def generate_data(num_obs, link=_identity_link, family=_gaussian_family,
                  return_mean=False, include_hft=False):
    purchases = [0, 3, 10, 16, 27, 30]
    ppurchases = [0.1, 0.2, 0.3, 0.3, 0.05, 0.05]
    genders = ['male', 'female']
    pgenders = [0.7, 0.3]
    countries = ['USA', 'CAN', 'GBR']
    np.random.seed(3)
    X = pd.DataFrame(data={'purchases': np.random.choice(purchases, size=num_obs, p=ppurchases),
                           'gender': np.random.choice(genders, size=num_obs, p=pgenders),
                           'country': np.random.choice(countries, size=num_obs),
                           'hft':     np.random.random(size=num_obs)
                           })
    gmu =  gmu_purchases(X['purchases'].values)
    gmu += gmu_gender(X['gender'].values)
    gmu += gmu_country(X['country'].values)
    if include_hft:
        gmu += gmu_hft(X['hft'].values)

    mu = link(gmu)
    if return_mean:
        y = mu
    else:
        y = family(mu)

    return X, y

def generate_covariate_class_data(return_mean=False):
    genders = ['male',
               'female',
               'male',
               'female',
               'male',
               'female',
               'male',
               'female',
               'male',
               'female',
               'male',
               'female',
               'male',
               'female',
               'male',
               'female',
               'male']

    countries = ['usa',
                 'usa',
                 'gbr',
                 'gbr',
                 'can',
                 'can',
                 'usa',
                 'usa',
                 'gbr',
                 'gbr',
                 'can',
                 'can',
                 'usa',
                 'usa',
                 'gbr',
                 'gbr',
                 'can']

    X = pd.DataFrame(data={'gender': genders,
                           'country': countries})

    ccs = np.array([1000, 1400, 2200, 1300, 3200, 1700,
                     500, 1700, 1400,  800, 2600, 1200,
                    1600,  900,  400, 1600, 1200])

    np.random.seed(4)
    gmu = gmu_gender(X['gender'].values)
    gmu += gmu_country(X['country'].values)
    mu = _logit_link(gmu)
    if return_mean:
        y = mu
    else:
        y = _binomial_family(mu, ccs)

    return X, y, ccs


def generate_spline_data(num_obs):
    X = pd.DataFrame(data={'hft': np.random.random(size=num_obs)})
    gmu = gmu_hft(X['hft'].values)
    mu = gmu
    y = _gaussian_family(mu)
    return X, y

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('variety', choices=['linear', 'logistic',
                                            'covariate', 'spline',
                                            'additive', 'cv'], help='Thing to test.')
    parser.add_argument('--save', action='store_true', help='Save model to file.')
    parser.add_argument('--load', action='store_true', help='Load model from file.')
    parser.add_argument('--plot', action='store_true', help='Plot convergence')
    args = parser.parse_args()

    if args.variety == 'linear':
        test_linear_regression(args.plot, args.save, args.load)
    elif args.variety == 'logistic':
        test_logistic_regression(args.plot, args.save, args.load)
    elif args.variety == 'covariate':
        test_logistic_regression_covariate_classes(args.plot, args.save, args.load)
    elif args.variety == 'spline':
        test_spline_regression(args.plot, args.save, args.load)
    elif args.variety == 'additive':
        test_additive_regression(args.plot, args.save, args.load)
    elif args.variety == 'cv':
        test_cross_validation()
