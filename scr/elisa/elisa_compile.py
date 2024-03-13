"""
05/10/2020: Version 0.1
04/08/2020: incorporating assymetrical fitting

"""
from elisa import ELISA
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import shutil
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
from lmfit import Model, Parameters, Minimizer, report_fit
import pybroom as br

from scipy.stats import stats

# import statsmodels.formula.api as smf
# from scipy.stats import zscore
# import re


class ELISACOMPILE(object):
    """
    Class to compile multiple elisa experiments into single analysis
    """

    def __init__(self, in_fl):
        """
        Initialize class global variables
        :param in_fl: (string) full path file name specifiying the excel files to be analyzed
        """
        self.in_fl = in_fl
        # Dataframes for each group of elisas (e.g. EBOV...). These dataframes get initialized everytime
        # we process a new group (e.g. BOMV)
        self.df = pd.DataFrame()
        self.df_zsc = pd.DataFrame()
        self.df_group = pd.DataFrame()
        self.df_models = pd.DataFrame()
        self.df_all = pd.DataFrame()
        self.df_all_models = pd.DataFrame()
        self.outdir = ""
        self.xlabel = ""
        self.ylabel = ""

    def read_infl(self):
        """
        Read the input file, analyze one file at a time, appending them and when reaching "//"
        process the data altogether
        :return:
        """
        cat = ""
        f = open(self.in_fl, "r")
        f1 = f.readlines()
        format_fl = 'nan'
        denoise = 0
        for ln in f1:
            ln = ln.rstrip()
            if ln[0] == "#":
                continue
            if ln == "//":
                # print(self.df)
                self.df_group = self.df.groupby(['Category', 'Specie'])
                self.df_zsc = self.calculate_zsc(self.df)
                self.print_df_2_xlxs(self.outdir + cat + '/' + cat + "_concat.xlsx", 'within_cat', normalize)
                self.plot_stdev(self.outdir + cat + '/' + cat + "_grouped_stdev.png")
                self.sigmoid_for_each_cat_sp_pair(self.outdir + cat + '/plot/', log_fh, normalize, ec50_fh, int(fitting_val), ec90_fh, ec10_fh, curvefit_fh, curvefitcons_fh, constrain_fit, equation)
                self.df_all_models = self.df_all_models.append(self.df_models)
                self.plot_sigmoid_together('within_cat', self.outdir + cat + '/' + cat + "_fitted_curves.png",
                                           self.outdir + cat + '/' + cat + "_predicted_readouts.csv",\
                                           self.outdir + cat + '/' + cat + "_auc.txt", equation)
                self.empty_df()
            ls = ln.split('\t')
            if len(ls) != 2:
                continue
            if ls[0] == "DR":
                self.outdir = ls[1]
                self.createdir(self.outdir)
                log_fh = open(self.outdir + 'processing.log', 'w')
                ec50_fh = open(self.outdir + 'ec50.txt', 'w')
                print('Sample', 'EC50', 'Notes', sep='\t', file=ec50_fh)
                ec90_fh = open(self.outdir + 'ec90.txt', 'w')
                print('Sample', 'EC90', 'Notes', sep='\t', file=ec90_fh)
                ec10_fh = open(self.outdir + 'ec10.txt', 'w')
                print('Sample', 'EC10', 'Notes', sep='\t', file=ec10_fh)
                curvefit_fh = open(self.outdir + 'curve_fit.txt', 'w')
                curvefitcons_fh = open(self.outdir + 'curve_fit_cons.txt', 'w')
            elif ls[0] == "CT":
                cat = ls[1]
                self.createdir(self.outdir + cat)
                self.createdir(self.outdir + cat + '/plot')
            elif ls[0] == "FL":
                dfx = self.process_eli_xp(ls[1], cat, format_fl, int(denoise), log_fh, int(normalize), colhead_logt)
                # df.to_excel('help.xlsx')
                self.df = self.df.append(dfx)
                self.df = self.sort_df_columns(self.df)
                self.df_all = self.df_all.append(dfx)
                self.df_all = self.sort_df_columns(self.df_all)
            elif ls[0] == "FT":
                format_fl = ls[1]
            elif ls[0] == "DE":
                denoise = ls[1]
            elif ls[0] == "NR":
                normalize = ls[1]
            elif ls[0] == "CL":
                if ls[1] == 'True':
                    colhead_logt = True
                else:
                    colhead_logt = False
            elif ls[0] == "SI":
                fitting_val = ls[1]
            elif ls[0] == "CO":
                constrain_fit = int(ls[1])
            elif ls[0] == "EQ":
                equation = ls[1].lower()
            elif ls[0] == "YL":
                self.ylabel = ls[1]
            elif ls[0] == "XL":
                self.xlabel = ls[1]
        self.plot_sigmoid_together('all_together', self.outdir + "all_fitted_curves.png",
                                   self.outdir + "all_predicted_readouts.csv", self.outdir + "auc.txt", equation)
        self.print_df_2_xlxs(self.outdir + "all_together_concat.xlsx", 'all_together', normalize)
        log_fh.close()
        ec50_fh.close()
        ec90_fh.close()
        ec10_fh.close()
        curvefit_fh.close()
        curvefitcons_fh.close()
        f.close()

    def empty_df(self):
        self.df = self.df.iloc[0:0, 0:0]
        self.df_zsc = self.df_zsc.iloc[0:0, 0:0]
        self.df_models = self.df_models.iloc[0:0, 0:0]

    @staticmethod
    def sort_df_columns(df):
        df_values1 = df.drop(['Input file', 'Category', 'Specie'], axis=1)
        df_names1 = df[['Input file', 'Category', 'Specie']]
        df_new1 = pd.DataFrame()
        # list = sorted( np.array(df_values.columns.values.tolist()) )
        listx = sorted(df_values1.columns.values.tolist())
        # print(listx)
        for item in listx:
            df_new1[item] = df_values1[item].copy()
        for item in df_names1.columns.values:
            df_new1[item] = df_names1[item].copy()
        # print(df_new)
        df_new1.reset_index(drop=True, inplace=True)
        # print(df_new)
        return df_new1

    @staticmethod
    def createdir(targetdir):
        if os.path.isdir(targetdir):
            shutil.rmtree(targetdir)
        os.mkdir(targetdir)

    def plot_sigmoid_together(self, mode, fl_fig, fl_csv, fl_auc, equation1):
        df, df_models = self.initialize_sigmoid(mode)
        col_min, col_max = self.find_min_max_colnames(df)
        # print('min: {}\tmax: {}'.format(col_min, col_max))
        x_arr_short = np.linspace(col_min, col_max, 200)
        x_arr_long = np.linspace(col_min, col_max, 5000)
        f = open(fl_csv, 'w')
        f_auc = open(fl_auc, 'w')
        print('Sample', 'AUC', sep=' ', file=f_auc)
        head = 'sample_name, ' + ','.join(['%.5f' % num for num in x_arr_short])
        print(head, file=f)
        plt.figure()
        fig, ax = plt.subplots()
        for index, row in df_models.iterrows():
            if equation1 == 'sym':
                sp, cat, min1, max1, ec50, hill = self.extract_fitted_params_from_rowdf(row)
                #PLOT
                ax.plot(x_arr_short, self.sigmoid_log_concentration(x_arr_short, min1, max1, ec50, hill), alpha=0.7,
                        linewidth=2, label='{} {}'.format(cat, sp))
                pred_y = np.array(self.sigmoid_log_concentration(x_arr_short, min1, max1, ec50, hill))
            elif equation1 == 'asym':
                try:
                    sp, cat, min1, max1, ec50, hill, s = self.extract_fitted_params_from_rowdf(row)
                except:
                    sp, cat, min1, max1, ec50, hill = self.extract_fitted_params_from_rowdf(row)
                    s = 1
                #PLOT
                ax.plot(x_arr_short, self.sigmoid_log_conc_asym(x_arr_short, min1, max1, ec50, hill, s), alpha=0.7,
                        linewidth=2, label='{} {}'.format(cat, sp))
                pred_y = np.array(self.sigmoid_log_conc_asym(x_arr_short, min1, max1, ec50, hill, s))
            pred_y_flatten = np.where(pred_y < 0, 0, pred_y)
            ln = cat + '_' + sp + ', ' + ','.join(['%.5f' % num for num in pred_y_flatten])
            print(ln, file=f)
            #AUC
            if equation1 == 'sym':
                pred_y = np.array(self.sigmoid_log_concentration(x_arr_long, min1, max1, ec50, hill))
            elif equation1 == 'asym':
                pred_y = np.array(self.sigmoid_log_conc_asym(x_arr_long, min1, max1, ec50, hill, s))
            pred_y_flatten = np.where(pred_y < 0, 0, pred_y)
            auc = np.trapz(pred_y_flatten,x_arr_long)
            print('{}_{}'.format(cat, sp), auc, file=f_auc)
        # plt.ylim(min_read, max_read)
        plt.title('Fitted curves')
        # plt.legend(fontsize=5, edgecolor='None')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.savefig(fl_fig, dpi=300)
        plt.close('all')
        f.close()
        f_auc.close()

    def initialize_sigmoid(self, mode):
        if mode == 'within_cat':
            df = self.df
            df_models = self.df_models
            print("Plotting sigmoidals within category")
        else:
            df = self.df_all
            df_models = self.df_all_models
            print("Plotting sigmoidals from all categories")
        return df, df_models

    @staticmethod
    def extract_fitted_params_from_rowdf(ln):
        sp = ln['Specie']
        cat = ln['Category']
        min1 = ln['min']
        max1 = ln['max']
        ec50 = ln['ec50']
        hill = ln['hill']
        return sp, cat, min1, max1, ec50, hill

    @staticmethod
    def find_min_max_colnames(df1):
        """
        Goes through the column names of a given dataframe, excludes those that are not numbers,
        returns the min and max values
        :param df1:
        :return:
        """
        ar = np.array([])
        for col_nm in df1.columns.values:
            if not isinstance(col_nm, float) and not isinstance(col_nm, int):
                continue
            ar = np.append(ar, col_nm)
        return np.min(ar), np.max(ar)

    def plot_stdev(self, fl):
        """
        plots the histogram for the standard deviation for each column (each column is a subplot)
        :param fl:
        :return:
        """
        df = self.df_group.std().reset_index()
#         print(df)
        df = df.drop(['Category', 'Specie'], axis=1)
        # print(df)
        # x_min = df.select_dtypes(include=['float64', 'int']).values.min()
        # x_max = df.values.max()
        # x_max = df.select_dtypes(include=['float64', 'int']).values.max()
        x_min = np.nanmin(df.values)
        x_max = np.nanmax(df.values)

        plt.figure()
        # print(x_min, x_max)
        print('Warning, the next #df.hist#elisa_compile.py#def plot_stdev# gives me a warning:')
        fig = df.hist(bins=50, range=(x_min, x_max), xlabelsize=8, ylabelsize=8, sharex=True, sharey=True)
        plt.subplots_adjust(hspace=0.7)
        plt.xticks(np.arange(0, x_max + 1, 20))
        for ax in fig.flatten():
            ax.set_xlabel("Standard deviation", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
        [x.title.set_size(10) for x in fig.ravel()]
        plt.savefig(fl, dpi=200)
        plt.close('all')

    def sigmoid_for_each_cat_sp_pair(self, plot_dir, fh, normalize_opt, ec50fh, fit_val_opt, ec90fh, ec10fh, curvefitfh, curvefitconsfh, constrainfit, equation1):
        """For each category (e.g. EBOV) and specie (e.g. B3), fit a sigmoid curve, evaluate the fitting, store
         the fitted parameters and make a separate plot with normalized
        replicate readouts and a sigmoid curved fitted using all points
        """
        specie_st = set(self.df.Specie)
        cat_st = set(self.df.Category)
        min_read, max_read = self.get_min_max_indataframe(self.df)
        if constrainfit == 2:
            max_read = 200
        self.createdir(plot_dir + '/sigmoidal')
        self.createdir(plot_dir + '/residuals')
        for cat in sorted(cat_st):
            residuals_dict = dict()
            for sp in sorted(specie_st):
                print("Sigmoid fitting for: {}_{}".format(cat, sp), file=fh)
                temp_df = self.df[self.df.Category.eq(cat) & self.df.Specie.eq(sp)]
                temp_df2 = temp_df.drop(['Input file', 'Category', 'Specie'], axis=1)
                print('Plotting sigmoidal fit for', cat, sp)
                # When fitting to the mean of the replicate data
                x_data, y_meandata, y_stddata = self.extract_mean_std_bycolumn(temp_df2)
                # When treating each replicate as an individual point
                x_data_all, y_data_all = self.extract_individual_points(temp_df2)
                # Setting initial parameter
                # ec50_init = self.estimate_init_ec50(x_data, y_meandata)
                chisq, pearson, residuals_ar, fit_min, fit_max, fit_ec50, fit_hill, fit_s = 0, 0, [], 0, 0, 0, 0, 1
                np.set_printoptions(precision=2)
                # Fitting
                try:
                    if fit_val_opt == 0:
                        # if equation1 == 'sym':
                        fit = curve_fit(self.sigmoid_log_concentration, x_data, y_meandata, sigma=y_stddata,
                                        method='lm', maxfev=10000)
                        # elif equation1 == 'asym':
                        #     fit = curve_fit(self.sigmoid_log_conc_asym, x_data, y_meandata, sigma=y_stddata,
                        #                     method='lm', maxfev=10000)
                    elif fit_val_opt == 1:
                        # if equation1 == 'sym':
                        fit = curve_fit(self.sigmoid_log_concentration, x_data_all, y_data_all,
                                        method='lm', maxfev=10000)
                        # elif equation1 == 'asym':
                        #     fit = curve_fit(self.sigmoid_log_conc_asym, x_data_all, y_data_all,
                        #                     method='lm', maxfev=10000)
                except:
                    print('\tWarning, Levenberg-Marquardt (lm) failed, trying dogbox')
                    if fit_val_opt == 0:
                        # if equation1 == 'sym':
                        fit = curve_fit(self.sigmoid_log_concentration, x_data, y_meandata,
                                        method='dogbox', maxfev=100000)
                        # elif equation1 == 'asym':
                        #     fit = curve_fit(self.sigmoid_log_conc_asym, x_data, y_meandata,
                        #                     method='dogbox', maxfev=10000)
                    elif fit_val_opt == 1:
                        # if equation1 == 'sym':
                        fit = curve_fit(self.sigmoid_log_concentration, x_data_all, y_data_all,
                                        method='dogbox', maxfev=100000)
                        # elif equation1 == 'asym':
                        #     fit = curve_fit(self.sigmoid_log_conc_asym, x_data_all, y_data_all,
                        #                     method='dogbox', maxfev=10000)
                # Extracting fitted parameters
                if str(fit[1][0][0]) != 'inf':
                    ans, cov = fit
                    # if equation1 == 'sym':
                    fit_min, fit_max, fit_ec50, fit_hill = ans
                    # elif equation1 == 'asym':
                    #     fit_min, fit_max, fit_ec50, fit_hill, fit_s = ans
                # Contrained fitting using the fitted parameters as initial parameters
                if constrainfit > 0:
                    # try:
                    pars = Parameters()
                    # elisa constrain
                    if constrainfit == 1:
                        pars.add('ymin', value=0, vary=False)
                        pars.add('ymax', value=fit_max, vary=True)
                        pars.add('ec50', value=fit_ec50, vary=True, min=0.0, max=fit_ec50+100)
                        pars.add('hill', value=fit_hill, vary=True, min=fit_hill-100, max=fit_hill+100)
                    # neut constrain
                    elif constrainfit == 2:
                        # print('ymin {:.2f}, ymax {:.2f}, ec50 {:.2f}, hill {:.2f}'.format(fit_min, fit_max, fit_ec50, fit_hill))
                        if fit_min < -100:
                            print('\tAdjusting for curve with fitted ymin < -100')
                            pars.add('ymin', value=0, vary=True, min=0.0, max=100.0)
                            pars.add('ec50', value=5, vary=True, min=0.00001)
                            pars.add('hill', value=1, vary=True, min=0.00001)
                            pars.add('ymax', value=100, vary=True, min=0.0, max=150.0)
                            if equation1 == 'asym':
                                pars.add('s', value=2, vary=True, min=0.00001)
                        else:
                            # print('No need to adjust for ymin')
                            pars.add('ymin', value=fit_min, vary=True, min=0.0, max=100.0)
                            pars.add('ec50', value=fit_ec50, vary=True, min=0.00001)
                            pars.add('hill', value=fit_hill, vary=True, min=0.00001)
                            pars.add('ymax', value=fit_max, vary=True, min=0.0, max=150.0)
                            if equation1 == 'asym':
                                pars.add('s', value=2, vary=True, min=0.00001)
                    if fit_val_opt == 0:
                        if equation1 == 'sym':
                            minner = Minimizer(self.residual_fun, pars, fcn_args=(x_data, y_meandata))
                        elif equation1 == 'asym':
                            minner = Minimizer(self.residual_fun_asym, pars, fcn_args=(x_data,
                                                                                       y_meandata))
                    else:
                        if equation1 == 'sym':
                            minner = Minimizer(self.residual_fun, pars, fcn_args=(x_data_all, y_data_all))
                        elif equation1 == 'asym':
                            minner = Minimizer(self.residual_fun_asym, pars, fcn_args=(x_data_all, y_data_all))
                    try:
                        result = minner.minimize()
                        fit_min = result.params['ymin'].value
                        fit_max = result.params['ymax'].value
                        fit_max = result.params['ymax'].value
                        fit_ec50 = result.params['ec50'].value
                        fit_hill = result.params['hill'].value
                        if equation1 == 'asym':
                            fit_s = result.params['s'].value
                    except:
                        print("\tWARNING, IT WAS NOT POSSIBLE TO TO FIT A CURVE, PARAMS NaN")
                        fit_min = np.nan
                        fit_max = np.nan
                        fit_max = np.nan
                        fit_ec50 = np.nan
                        fit_hill = np.nan
                        if equation1 == 'asym':
                            fit_s = np.nan
                    # except:
                    #     print('Warning skipping constraint fit for '+ sp)
                print()
                if equation1 == 'sym':
                    print('\tFitted params (min, max, ec50, hill): {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(fit_min, fit_max, fit_ec50, fit_hill), file=fh)
                    print('{}_{}'.format(cat, sp), fit_min, fit_max, fit_ec50, fit_hill, sep='\t', file=curvefitfh)
                elif equation1 == 'asym':
                    print('\tFitted params (min, max, ec50, hill, s): {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(fit_min, fit_max, fit_ec50, fit_hill, fit_s), file=fh)
                    print('{}_{}'.format(cat, sp), fit_min, fit_max, fit_ec50, fit_hill, fit_s, sep='\t', file=curvefitfh)
                # Calculating EC90 and EC10 (I am not sure if the formula need to be different when asymmetrical)
                f = 90
                try:
                    fit_ec90 = ( (f/(100-f))**(1/fit_hill) )* fit_ec50
                except:
                    fit_ec90 = 0
                f = 10
                try:
                    fit_ec10 = ((f / (100 - f)) ** (1 / fit_hill)) * fit_ec50
                except:
                    fit_ec10 = 0
                # Print out ECx
                if fit_ec50 < 10**x_data.max():
                    print('\tEC50 fitted:', '{:.9f}'.format(fit_ec50), file=fh)
                    print('{}_{}'.format(cat, sp), '{:.9f}'.format(fit_ec50), file=ec50fh, sep='\t')
                    print('{}_{}'.format(cat, sp), '{:.9f}'.format(fit_ec90), file=ec90fh, sep='\t')
                    print('{}_{}'.format(cat, sp), '{:.9f}'.format(fit_ec10), file=ec10fh, sep='\t')

                else:
                    print('\tEC50 fitted:', '{:.9f}'.format(fit_ec50), 'Warning!_EC50_is_out_of_experimental_x-range', file=fh)
                    print('{}_{}'.format(cat, sp), '{:.9f}'.format(fit_ec50), 'Warning!_EC50_is_out_of_experimental_x-range', file=ec50fh, sep='\t')
                    print('{}_{}'.format(cat, sp), '{:.9f}'.format(fit_ec90), 'Warning!_EC50_is_out_of_experimental_x-range', file=ec90fh, sep='\t')
                    print('{}_{}'.format(cat, sp), '{:.9f}'.format(fit_ec10), 'Warning!_EC50_is_out_of_experimental_x-range', file=ec10fh, sep='\t')
                # Evaluate fitting
                chisq, pearson, pval, residuals_ar, r2 = self.evaluate_fit(x_data, y_meandata,
                                                                               y_stddata, fit_min, fit_max,
                                                                               fit_ec50, fit_hill, fit_s, equation1)
                # Print Evaluation
                print('\tChisq', '{:.4f}'.format(chisq), file=fh, sep='\t')
                print('\tPearson: {:.4f}\t{:.2E}'.format(pearson, pval), file=fh, sep='\t')
                print('\tR squared:', '{:.4f}'.format(r2), file=fh, sep='\t')
                if pearson == 'nan' or pearson < 0.95 or r2 < 0.9:
                    s = '\tWARNING! something wrong when fitting the sigmoid (Pearson < 0.95 and/or R2 < 0.9).\n\tPearson: {:.3f}\n\tR2: {:.3f}'.format(pearson, r2)
                    print(s)
                    print('\tWARNING! something wrong when fitting the sigmoid (Pearson < 0.95 and/or R2 < 0.9)', file=fh, sep='\t')
                if constrainfit > 0 and 'result' in locals():
                    print('\tConstrained fit results & evaluation', file=fh)
                    dg = br.glance(result)
                    print(dg, file=fh)
                    dt = br.tidy(result)
                    print(dt, file=fh)
                    del result
                # Calculate residuals
                nm = '{}_{}'.format(cat, sp)
                if nm not in residuals_dict.keys():
                    residuals_dict[nm] = {}
                residuals_dict[nm]['x_values'] = x_data
                residuals_dict[nm]['residuals'] = residuals_ar
                residuals_dict[nm]['stddev'] = y_stddata
                # Save models for later where I will be plotting all models together
                if equation1 == 'sym':
                    self.df_models = self.df_models.append({'min': fit_min, 'max': fit_max, 'ec50': fit_ec50,
                                                            'hill': fit_hill, 'Specie': sp, 'Category': cat},
                                                           ignore_index=True)
                elif equation1 == 'asym':
                    self.df_models = self.df_models.append({'min': fit_min, 'max': fit_max, 'ec50': fit_ec50,
                                                            'hill': fit_hill, 'Specie': sp, 'Category': cat,
                                                            's': fit_s}, ignore_index=True)
                # Plotting
                self.plot_separate(temp_df, plot_dir + '/sigmoidal', cat, sp, x_data, y_meandata,
                                   y_stddata, fit_min, fit_max, fit_ec50, fit_hill, chisq, min_read, max_read,
                                   pearson, r2, fit_s, equation1)
            # Plotting residuals using the same range for the y axis
            self.plot_residuals(plot_dir + '/residuals', residuals_dict)

    @staticmethod
    def plot_residuals(pltdir, dc):
        """
        Plots residuals as described in a dictionary using the same range of values for the y-axis
        :param pltdir: directory where to save the plots
        :param dc: dictionary[name]['residuals'] = np.array | dictionary[name]['stddev'] = np.array
        :return:
        """
        y_min, y_max = 1000000, 0
        for nm in dc.keys():
            # print('Plotting residuals for {}'.format(nm))
            pos = 0
            for y_val in dc[nm]['residuals']:
                lowest = y_val - dc[nm]['stddev'][pos] - 5
                highest = y_val + dc[nm]['stddev'][pos] + 5
                if lowest < y_min:
                    y_min = lowest
                if highest > y_max:
                    y_max = highest
                pos += 1
        for nm in dc.keys():
            fl = pltdir + "/" + nm + "_residual.png"
            plt.figure()
            plt.errorbar(dc[nm]['x_values'], dc[nm]['residuals'], dc[nm]['stddev'], fmt='.', mfc='blue', mec='black',
                         capsize=3, elinewidth=0, ecolor='black')
            plt.ylim(y_min, y_max)
            plt.hlines(0, np.min(dc[nm]['x_values']), np.max(dc[nm]['x_values']))
            plt.title(nm)
            plt.xlabel("log10[rVSV]")
            plt.ylabel("Residuals (observed - predicted)")
            plt.savefig(fl, dpi=200)
            plt.close()

    def plot_separate(self, temp_df, plot_dir, cat, sp, x_data, y_meandata, y_stddata,
                      fit_min, fit_max, fit_ec50, fit_hill, chisq, min_read, max_read, pearson, r2, fit_s, equation2):
        """
        For any given cat_sp pair (e.g EBOV B12) this function will generate a plot containing the readouts for
        the different replicates, the corresponding average readouts along with its standard deviations and the
        fitted sigmoidal curve. Also, evaluation parameters (chisq and pearson) will also be included
        :param temp_df: dataframe with the readouts for the specific cat and specie
        :param plot_dir: directory were to save the plot
        :param cat: category (e.g. EBOV)
        :param sp: specie (e.g. B12)
        :param x_data: numpy array with the x-coordinate of the readouts, the concentration
        :param y_meandata: numpy array with the y coordinate of the readouts, the signal
        :param y_stddata: numpy array with the std deviation of the readouts
        :param fit_min: fitted minimum for sigmoidal
        :param fit_max: fitted maximum for sigmoidal
        :param fit_ec50: fitted ec50 for sigmoidal
        :param fit_hill: fitted hill coeficient for sigmoidal
        :param chisq: chi-square
        :param min_read: min readout value of the entire dataframe (containing all species for a given category
                        this is used to have the same y-range in different plots
        :param max_read: max readout value of the entire dataframe (containing all species for a given category
                        this is used to have the same y-range in different plots
        :param pearson: pearson correlation
        :return:
        """
        min_x = np.min(x_data)
        max_x = np.max(x_data)
        t = np.linspace(min_x - 1, max_x, 200)
        plot_fl = plot_dir + "/" + cat + "_" + sp + "_sigmoidal.png"

        plt.figure()
        fig, ax = plt.subplots()
        plt.errorbar(x_data, y_meandata, y_stddata, fmt='.', mfc='black', mec='black',
                     label='Mean', alpha=0.4, capsize=3, elinewidth=1, ecolor='black')
        # Plotting the normalized data
        # Mode 1
        # temp_dfgroup = temp_df.groupby('Input file')
        # df_t = self.transpose_df(temp_df, 'Input file', ['Input file', 'Category', 'Specie'], 0)
        # for name, group in temp_dfgroup:
        #     ax.plot(df_t[name], marker='o', linestyle='-', ms=3, label=name, alpha=0.7)
        # Mode 2 (deals with missing data)
        column_list = np.array(temp_df.columns.values.tolist()[:-3])
        for index, row in temp_df.iterrows():
            name = row.values.tolist().pop(-3)
            val_list = np.array(row.values.tolist()[:-3])
            mask = np.isfinite(val_list)
            ax.plot(column_list[mask], val_list[mask], marker='o', linestyle='-', ms=3, label=name, alpha=0.7)
        # Plotting the sigmoidal
        if equation2 == 'sym':
            plt.plot(t, self.sigmoid_log_concentration(t, fit_min, fit_max, fit_ec50, fit_hill), label='fitted',
                     alpha=0.7, color='black', linewidth=2)
        elif equation2 == 'asym':
            plt.plot(t, self.sigmoid_log_conc_asym(t, fit_min, fit_max, fit_ec50, fit_hill, fit_s), label='fitted',
                     alpha=0.7, color='black', linewidth=2)
        plt.figtext(0.15, 0.16, 'chi-sq: %.2f' % chisq, fontsize=5)
        plt.figtext(0.15, 0.14, 'Pearson: %.2f' % pearson, fontsize=5)
        plt.figtext(0.15, 0.12, 'R\u00b2: %.2f' % r2, fontsize=5)
        plt.ylim(min_read, max_read)
        plt.title(cat + "-" + sp)
        plt.legend(fontsize=4, edgecolor='None')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.savefig(plot_fl, dpi=200)
        plt.close('all')

    def evaluate_fit(self, exp_x, exp_y, exp_uncertain, f_min, f_max, f_ec50, f_hill, f_s, equation2):
        """
        Evaluates the fitted curve compared to the observed data. It calculates chisq, pearson, residuals
        :param exp_x: np.array independent variable (concentration) for the x-axis
        :param exp_y: np.array dependent variable (observed mean absorbance at given concentrations)
        :param exp_uncertain: np.array standard deviation
        :param f_min: fitted_min for sigmoidal
        :param f_max: fitted_max for sigmoidal
        :param f_ec50: fitted_ec50 for sigmoidal
        :param f_hill: fitted_hill for sigmoidal
        :return:
            chisquare, pearson, residuals_ara
        """
        chisq = 0
        pos = 0
        residuals_ar = np.array([])
        pred_y_ar = np.array([])
        for item in exp_y:
            y_obs = item
            x_val = exp_x[pos]
            u_val = exp_uncertain[pos]
            if equation2 == 'sym':
                y_pred = self.sigmoid_log_concentration(x_val, f_min, f_max, f_ec50, f_hill)
            elif equation2 == 'asym':
                y_pred = self.sigmoid_log_conc_asym(x_val, f_min, f_max, f_ec50, f_hill, f_s)
            residuals_ar = np.append(residuals_ar, y_obs - y_pred)
            pred_y_ar = np.append(pred_y_ar, y_pred)
            chisq1 = ((y_obs - y_pred)/u_val)**2
            chisq += chisq1
            # print('\tconcentration\t', x_val,   '\tobserved\t', y_obs, '\tpredicted\t', '{:.2f}'.format(y_pred),
            #       '\tuncertainty\t', u_val, '\tpoint_chisq\t','{:.2f}'.format(chisq1),
            #       '\tcumul_sq\t', '{:.2f}'.format(chisq))
            pos += 1
        # chisq = sum(((exp_y - self.sigmoid2(exp_x, f_min, f_max, f_ec50, f_hill)) / exp_uncertain) ** 2)
        # print(exp_y,'\t',pred_y_ar)
        try:
            pearson, pval = pearsonr(exp_y, pred_y_ar)
            r2 = r2_score(exp_y, pred_y_ar)
        except:
            pearson, pval, r2 = 0, 1, 0
        # slope, intercept, r_val, p_val, std_err = stats.linregress(exp_y, pred_y_ar)
        # r22 = r_val**2
        # print(r2,r22)
        # print('\t', 'chisq', '{:.2f}'.format(chisq))
        # print('\t', 'pearson', '{:.2f}'.format(pearson), '{:.2E}'.format(pval))
        return chisq, pearson, pval, residuals_ar, r2

    @staticmethod
    def estimate_init_ec50(x_ar, y_ar):
        """
        Estimates the approximae EC50 value using only the readouts, this is useful as an initial estimator
        in order to estimate the parameters of the sigmoidal curve
        :param x_ar: array listing the concentrations
        :param y_ar: array listing the readouts
        :return:
        """
        half_value_target, dif = np.max(y_ar) / 2, 1000
        ct = 0
        pos = 0
        for item in y_ar:
            if abs(half_value_target - item) < dif:
                dif = abs(half_value_target - item)
                pos = ct
            ct += 1
        ec50_init = x_ar[pos]
        return ec50_init

    @staticmethod
    def get_min_max_indataframe(df):
        # max_val = df.select_dtypes(include=['float64', 'int']).values.max()
        # min_val = df.select_dtypes(include=['float64', 'int']).values.min()
        # print(df)
        # print(df.iloc[:, :-3])
        # max_val = np.nanmax(df.iloc[:, :-3].values)
        # min_val = np.nanmin(df.iloc[:, :-3].values)
        # df.to_csv('test.csv')
        df_temp = df.drop(['Input file', 'Category', 'Specie'], axis=1)
        max_val = np.nanmax(df_temp.values)
        min_val = np.nanmin(df_temp.values)

        return min_val, max_val

    @staticmethod
    def extract_individual_points(df):
        x_ar = np.array([])
        y_ar = np.array([])
        # y_ar = np.append(y_ar, np.array(df[column]))
        # x_ar = np.append(x_ar, float(column))
        # print(df)
        for indx,row in df.iterrows():
            yar1 = np.array(df.loc[indx, :])
            xar1 = np.array(df.columns.tolist())
            yar2 = yar1[np.logical_not(np.isnan(yar1))]
            xar2 = xar1[np.logical_not(np.isnan(yar1))]
            y_ar = np.append(y_ar, yar2)
            x_ar = np.append(x_ar, xar2)
        # print(x_ar)
        # print(y_ar)
        return x_ar, y_ar

    @staticmethod
    def extract_mean_std_bycolumn(df):
        """
        For a given dataframe, it goes column by column, for each column:
            calculates the mean and std dev
            saves the value of the column name (it has to be float or integer) as x coordinate in an np array
            saves the mean as y coordinate in an np array
            saves the std as uncertainty in an np array
        These three np arrays will be used to fit a sigmoidal
        :param df:
        :return:
            x_ar: np array
            y_mean_ar: np array
            y_std_ar: np_array
        """
        x_ar = np.array([])
        y_mean_ar = np.array([])
        y_std_ar = np.array([])
        for column in df:
            y_ar = np.array(df[column])
            if np.count_nonzero(~np.isnan(y_ar)) == 0:
                continue
            y_mean = float(format(np.nanmean(y_ar), '.2f'))
            y_std = float(format(np.nanstd(y_ar), '.2f'))
            # print('ymean is ', y_mean)
            y_mean_ar = np.append(y_mean_ar, y_mean)
            y_std_ar = np.append(y_std_ar, y_std)
            x_ar = np.append(x_ar, float(column))

        # if standard deviation equals 0, give a dummy value (otherwise the fitting gets wrong)
        y_std_ar[y_std_ar == 0] = 0.1
        return x_ar, y_mean_ar, y_std_ar

    @staticmethod
    def sigmoid_log_concentration(x, ymin, ymax, ec50, hill):
        """
        Sigmoid equation used in PRISM when the concentration of the antigen is in the log scale
        https://www.graphpad.com/guides/prism/8/curve-fitting/reg_dr_stim_variable.htm
        :param x: Log concentration value OR numpy array with Log_concentration values
        :param ymin: minimum readout
        :param ymax: maximum readout
        :param ec50: ec50
        :param hill: hill coefficient
        :return:
            y : predicted readout for the x value/np.array
        """
        warnings.filterwarnings("ignore")
        # y = ymin + (ymax - ymin) / (1 + 10 ** ((np.log(ec50) - x) * hill))
        y = ymin + (ymax - ymin) / (1 + 10 ** ((np.log10(ec50) - x) * hill))
        # y = ymin + (ymax - ymin) / (1 + 10 ** ((np.log10(np.abs(ec50)) - x) * hill))
        warnings.filterwarnings("default")
        return y

    @staticmethod
    def sigmoid_log_conc_asym(x, ymin, ymax, ec50, hill, s):
        # https: // www.graphpad.com / guides / prism / latest / curve - fitting / reg_asymmetric_dose_response_ec.htm
        logXb = np.log10(ec50) + (1/hill)*np.log10((2**(1/s))-1)
        numerator = ymax - ymin
        denominator = (1 + 10**((logXb-x)*hill))**s
        y = ymin + (numerator/denominator)
        return y

    def residual_fun(self, params, x, data):
        ymin = params['ymin']
        ymax = params['ymax']
        ec50 = params['ec50']
        hill = params['hill']
        ypred = self.sigmoid_log_concentration(x, ymin, ymax, ec50, hill)
        return (ypred - data)

    def residual_fun_asym(self, params, x, data):
        ymin = params['ymin']
        ymax = params['ymax']
        ec50 = params['ec50']
        hill = params['hill']
        s = params['s']
        ypred = self.sigmoid_log_conc_asym(x, ymin, ymax, ec50, hill, s)
        return(ypred - data)

    @staticmethod
    def sigmoid_concentration(x, ymin, ymax, ec50, hill):
        """
        Sigmoid equation used in PRISM when the concentration of the antigen is in NOT in the log scale
        https://www.graphpad.com/guides/prism/8/curve-fitting/reg_dr_stim_variable_2.htm
        :param x: concentration value OR numpy array with concentration values
        :param ymin: minimum readout
        :param ymax: maximum readout
        :param ec50: ec50
        :param hill: hill coefficient
        :return:
            y : predicted readout for the x value/np.array
        """
        y = ymin + x**hill * (ymax - ymin) / (x**hill + ec50**hill)
        return y

    # def plot_polyn_separate(self, plot_dir):
    #     """For each category (e.g. EBOV) and specie (e.g. B3) make a separate plot with normalized
    #     replicate readouts and a polynomical curve fitted using all points
    #     """
    #     specie_st = set(self.df.Specie)
    #     cat_st = set(self.df.Category)
    #     max_read = self.df.select_dtypes(include=['float64']).values.max()
    #     min_read = self.df.select_dtypes(include=['float64']).values.min()
    #     # print("MAX READOUT IS {}".format(max_read))
    #     for cat in cat_st:
    #         for sp in specie_st:
    #             temp_df = self.df[self.df.Category.eq(cat) & self.df.Specie.eq(sp)]
    #             temp_dfgroup = temp_df.groupby('Input file')
    #             plot_fl = plot_dir + "/" + cat + "_" + sp + "_polynomial.png"
    #             df_t = self.transpose_df(temp_df, 'Input file', ['Input file', 'Category', 'Specie'], 0)
    #             model, min_x, max_x = self.fit_polynomial(temp_df, 4, plot_dir, cat, sp)
    #             self.df_models = self.df_models.append({'Model': model, 'Specie': sp,
    #                                                    'Category': cat}, ignore_index=True)
    #             t = np.linspace(min_x, max_x, 200)
    #             # EXTRAPOLATE POINTS IN FITTED CURVE
    #             # x_new = 1
    #             # y_new = model(x_new)
    #             # print('value x {} maps to value y {}'.format(x_new, y_new))
    #             plt.figure()
    #             fig, ax = plt.subplots()
    #             for name, group in temp_dfgroup:
    #                 ax.plot(df_t[name], marker='o', linestyle='-', ms=3, label=name, )
    #             ax.plot(t, model(t), linestyle='-', label='Fitted curve', color='black', linewidth=2, alpha=0.7)
    #             plt.ylim(min_read, max_read)
    #             plt.title(cat+"-"+sp)
    #             plt.legend(fontsize=4, edgecolor='None')
    #             plt.xlabel("-log10[rVSV]")
    #             plt.ylabel("Elisa signal (450nm)")
    #
    #             #plt.savefig(plot_fl, dpi=200)
    #             plt.close('all')
    #
    # @staticmethod
    # def fit_polynomial(df, degree, pltdir, cat, sp):
    #     """
    #     Fit a polynomial curve to dataframe and evaluate the fit
    #     :param df: dataframe
    #     :param degree: degree of the polynomial
    #     :param pltdir: directory where to save each plot
    #     :param cat: category (EBOV, SUDV, BOMV...)
    #     :param sp: specie (B1, B2...)
    #     :return:
    #         model: fitted polynomial model
    #         min_local: minimum concentration for which there are readouts
    #         max_local: maximum concentration for which there are readouts
    #     """
    #     # eval_fl = pltdir + "/" + cat + "_" + sp + "_curvefit_evaluation.txt"
    #     x = []
    #     y = []
    #     min_local = 1000
    #     max_local = 0
    #     for col_name in df:
    #         if type(col_name) == str:
    #             continue
    #         # print("column name is {} and type is {}".format(col_name, type(col_name)))
    #         if col_name < min_local:
    #             min_local = col_name
    #         if col_name > max_local:
    #             max_local = col_name
    #         for val in df[col_name]:
    #             if type(val) == str:
    #                 continue
    #             x.append(col_name)
    #             y.append(val)
    #             # print("\tvalue is {}".format(val))
    #     weights = np.polyfit(x, y, degree)
    #     model = np.poly1d(weights)
    #     # Evaluate model fitting
    #     df_xy = pd.DataFrame(columns=['y', 'x'])
    #     df_xy['x'] = x
    #     df_xy['y'] = y
    #     fit_eval = smf.ols(formula='y ~ model(x)', data=df_xy).fit()
    #     if fit_eval.rsquared < 0.75:
    #         print("Potential poor fitting for {}-{}, rsquared: {:.3f}".format(cat, sp, fit_eval.rsquared))
    #     # sample = open(eval_fl, 'w')
    #     # print(fit_eval.summary(), file=sample)
    #     # sample.close()
    #     return model, min_local, max_local

    @staticmethod
    def transpose_df(df, col_name, drop_ls, axis_numb):
        """
        Transpose a dataframe and remove specified columns in the transposed dataframe
        :param df: dataframe
        :param col_name: column specifying the name for all other columns in the transposed dataframe
        :param drop_ls: list of column names to be deleted
        :param axis_numb: dataframe axis
        :return:
        """
        df_t = df.transpose()
        df_t.columns = df_t.loc[col_name]
        for nm in drop_ls:
            df_t = df_t.drop(nm, axis=axis_numb)
        return df_t

    @staticmethod
    def calculate_zsc(df):
        """
        Calculates the z-score in a dataframe
        :param df: dataframe
        :return: df_zall dataframe with zscores
        """
        df_zall = pd.DataFrame()
        specie_st = set(df.Specie)
        cat_st = set(df.Category)
        for cat in cat_st:
            for sp in specie_st:
                temp_df = df[df.Category.eq(cat) & df.Specie.eq(sp)]
                numeric_cols = temp_df.select_dtypes(include=['float64', 'int']).columns
                # z_df = temp_df[numeric_cols].apply(zscore)
                z_df = (temp_df[numeric_cols] - temp_df[numeric_cols].mean()) / temp_df[numeric_cols].std()
                z_df['Specie'] = pd.Series(temp_df['Specie'])
                z_df['Category'] = pd.Series(temp_df['Category'])
                z_df['Input file'] = pd.Series(temp_df['Input file'])
                df_zall = df_zall.append(z_df)
        return df_zall

    def process_eli_xp(self, fl, cat_in, format_fl, denoise_opt, fh, normalize_opt, colhead_logt):
        """
        Calls the ELISA class to process one particular excel file and returns the corresponding normalized
        dataframe
        :param fl: excel file
        :param cat_in: category (EBOV, SUDV...)
        :param format_fl: file specifiying the type of format in the excel file
        :param denoise_opt: denoise option, either 1, 2, 3
        :return: df normalized dataframe
        """
        # print('denoise', denoise_opt)

        elisa_xp = ELISA(fl, format_fl, self.outdir + cat_in, denoise_opt, normalize_opt, colhead_logt)
        elisa_xp.process_fl()
        # print('joder0')

        if normalize_opt == 0:
            df_t = elisa_xp.df_denoised
        elif normalize_opt >=1:
            df_t = elisa_xp.df_normal

        df_t['Category'] = cat_in
        df_t['Input file'] = elisa_xp.basename
        print("Processing\t{}".format(fl), file=fh)

        del elisa_xp
        # print(elisa_xp)
        return df_t

    def print_df_2_xlxs(self, fl, mode, normalize_opt):
        """
        Prints into an excel file the dataframe containing all readouts belonging to a particular
        category (e.g. EBOV)
        :param fl: full path for filename
        :return:
        """
        # print('FILE IS', fl)
        df = pd.DataFrame()
        df_group = pd.DataFrame()
        if mode == 'within_cat':
            df = self.df
            df_group = self.df_group
        else:
            df = self.df_all
            df_group = df.groupby(['Category', 'Specie'])
        ds = df_group.size()

        # print(ds)
        alp_s = 'A-B-C-D-E-F-G-H-I-J-K-L-M-N-O-P-Q-R-S-T-U-V-W-X-Y-Z-AA-AB'
        alp_ls = alp_s.split('-')
        with pd.ExcelWriter(fl) as writer:
            specie_ct = len(set(df.Specie))
            group_ct = len(set(df.Category))
            row_ct = specie_ct * group_ct

            workbook = writer.book
            # Add a format. Light red fill with dark red text.
            format_red = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
            # Add a format. Green fill with dark green text.
            format_green = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})

            # Print count
            worksheet = workbook.add_worksheet('Count')
            writer.sheets['Count'] = worksheet
            worksheet.write_string(0, 0, 'Number of replicates')
            ds.to_excel(writer, sheet_name='Count', startrow=1)
            worksheet.write_string('C2', 'Count')
            cell_range = 'C3:C' + str(2 + row_ct)
            worksheet.autofilter('A2:C2')
            worksheet.conditional_format(cell_range, {'type': 'cell', 'criteria': '<',
                                                      'value': ds.median(), 'format': format_red})
            # Print normalized
            worksheetname = 'joder'
            if int(normalize_opt) == 0:
                worksheetname = 'Denoised'
            elif int(normalize_opt) == 1:
                worksheetname = 'Normalized'
            worksheet = workbook.add_worksheet(worksheetname)
            writer.sheets[worksheetname] = worksheet
            worksheet.write_string(0, 0, 'Concatenated ' + worksheetname +' readouts')
            col_ct = df.shape[1]
            col_let = alp_ls[col_ct-3]
            df.to_excel(writer, sheet_name=worksheetname, startrow=1)
            cell_range = 'B3:' + col_let + str(2+df.shape[0])
            worksheet.autofilter('B2:' + alp_ls[col_ct] + '2')
            max = 100
            if int(normalize_opt) == 0 :
                max = np.max(np.array(df.max(axis=0, numeric_only=True)))
                # print('max is ', max)
                worksheet.conditional_format(cell_range, {'type': '3_color_scale', 'min_type': 'num',
                                                      'max_type': 'num', 'min_value': 0, 'max_value': max})
            else:
                worksheet.conditional_format(cell_range, {'type': '3_color_scale', 'min_type': 'num',
                                                          'max_type': 'num', 'min_value': 0, 'max_value': 100})
            # Print Z-score
            if (mode == 'within_cat'):
                worksheet = workbook.add_worksheet('Z-score')
                writer.sheets['Z-score'] = worksheet
                worksheet.write_string(0, 0, 'Concatenated z-scores')
                self.df_zsc.to_excel(writer, sheet_name="Z-score", startrow=1)
                cell_range = 'B3:' + col_let + str(2+self.df_zsc.shape[0])
                worksheet.autofilter('B2:' + alp_ls[col_ct] + '2')
                worksheet.conditional_format(cell_range, {'type': 'cell', 'criteria': 'between',
                                                          'minimum': -2, 'maximum': -3, 'format': format_green})
                worksheet.conditional_format(cell_range, {'type': 'cell', 'criteria': 'between',
                                                          'minimum': 2, 'maximum': 3, 'format': format_green})
                worksheet.conditional_format(cell_range, {'type': 'cell', 'criteria': '<=',
                                                          'value': -3, 'format': format_red})
                worksheet.conditional_format(cell_range, {'type': 'cell', 'criteria': '>=',
                                                          'value': 3, 'format': format_red})
            # Print mean
            worksheet = workbook.add_worksheet('Mean')
            writer.sheets['Mean'] = worksheet
            worksheet.write_string(0, 0, 'Mean')
            df_group.mean().to_excel(writer, sheet_name="Mean", startrow=1)
            cell_range = 'C3:' + alp_ls[col_ct] + str(row_ct + 2)
            if int(normalize_opt) == 0:
                worksheet.conditional_format(cell_range, {'type': '3_color_scale', 'min_type': 'num',
                                                      'max_type': 'num', 'min_value': 0, 'max_value': max})
            else:
                worksheet.conditional_format(cell_range, {'type': '3_color_scale', 'min_type': 'num',
                                                      'max_type': 'num', 'min_value': 0, 'max_value': 100})
            worksheet.write_string(row_ct + 4, 0, 'Std.dev')
            df_group.std().to_excel(writer, sheet_name="Mean", startrow=row_ct + 5)
            worksheet.write_string(row_ct + 4 + row_ct + 4, 0, 'Std.dev / mean Std.Dev')
            dfx = df_group.std() / df_group.std().mean()
            dfx.to_excel(writer, sheet_name="Mean", startrow=row_ct + row_ct + 9)
            cell_range = 'C' + str(2*row_ct + 11) + ':' + alp_ls[col_ct] + str(3*row_ct + 10)
            worksheet.conditional_format(cell_range, {'type': 'cell', 'criteria': 'between',
                                                      'minimum': 2, 'maximum': 3, 'format': format_green})
            worksheet.conditional_format(cell_range, {'type': 'cell', 'criteria': '>',
                                                      'value': 3, 'format': format_red})

    # def remove_neg(self):
    #     """
    #     Removes row in the dataframe where the Specie value starts with "NEG"
    #     :return:
    #     """
    #     filt = self.df.Specie.str.contains("^NEG|^\*")
    #     df_temp = self.df[~filt]
    #     self.df = df_temp


if __name__ == '__main__':
    # setting tab-delimited file pointing to excel files
    ebov_fl = 'COVID_files.txt'
    elisa_comp = ELISACOMPILE(ebov_fl)
    elisa_comp.read_infl()
