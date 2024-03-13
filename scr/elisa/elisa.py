"""
05/10/2020: Version 0.1
Currently working on version 0.2
    it parses xls sheets into four different dataframes: i) experimental readouts; ii) positive control readouts
    iii) negative control readouts iv) blank readouts
"""
import os
import re
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys


class ELISA(object):
    """Class to process cytation experiments results described in an excel file. Processing includes:
    1. Denoising
        For each protein tested in an Elsa plate: average the replicate numbers at a given concentration
        and identify the lowest average value. This value will be considered as the background noise (each
        tested protein will have its own background noise). The raw readout will be denoised by substracting
        the corresponding background noise
    2. Normalizing
        Using a control protein as reference, we first identify the maximum average readout after denoising
        for the control protein. Then we normalize: denoised readout / max control average readout * 100
    3. Save processed data and plots
        the raw, denoised and normalized readouts are saved in an excel file so that we can trace back how the
        processing was done. Corresponding plots are also saved
        Excel file with raw and process data and plots will be save in a directory with the same name as the
        input excel file (the directory is saved in the same location as the input excel file)
    """

    def __init__(self, in_xls_fl, format_fl1, outdir, denoise_opt, normalize_opt, colhead_logt):
        self.df_denoised = pd.DataFrame()
        self.df_normal = pd.DataFrame()
        print("Reading excel:\t{}".format(in_xls_fl))
        self.basename = self.get_basename(in_xls_fl)
        if outdir == '':
            self.outdir, self.out_xls_fl = self.out_from_infl(in_xls_fl)
        else:
            self.outdir, self.out_xls_fl = self.out_from_user(outdir)
        self.in_xls = pd.ExcelFile(in_xls_fl)
        self.writer = pd.ExcelWriter(self.out_xls_fl, engine='xlsxwriter')
        self.workbook = self.writer.book
        self.xls_coords = {}
        # self.populate_xls_coords()
        self.read_format_fl(format_fl1)
        self.denoise_option = denoise_opt
        self.normalize_option = normalize_opt
        self.colhead_logt = bool(colhead_logt)

    # def populate_xls_coords(self):
    #     self.xls_coords['experiment'] = {}
    #     self.xls_coords['experiment']['readout'] = {}
    #     self.xls_coords['experiment']['readout']['usecols'] = 'B:M'
    #     self.xls_coords['experiment']['readout']['skiprows'] = 8
    #     self.xls_coords['experiment']['readout']['nrows'] = 8
    #     self.xls_coords['experiment']['concentration'] = {}
    #     self.xls_coords['experiment']['concentration']['usecols'] = 'B:M'
    #     self.xls_coords['experiment']['concentration']['skiprows'] = 6
    #     self.xls_coords['experiment']['concentration']['nrows'] = 1
    #     self.xls_coords['experiment']['name'] = {}
    #     self.xls_coords['experiment']['name']['usecols'] = 'N:N'
    #     self.xls_coords['experiment']['name']['skiprows'] = 8
    #     self.xls_coords['experiment']['name']['nrows'] = 8
    #     self.xls_coords['experiment']['keyword'] = 'nan'
    #
    #     self.xls_coords['positive'] = {}
    #     self.xls_coords['positive']['readout'] = {}
    #     self.xls_coords['positive']['readout']['usecols'] = 'B:M'
    #     self.xls_coords['positive']['readout']['skiprows'] = 8
    #     self.xls_coords['positive']['readout']['nrows'] = 8
    #     self.xls_coords['positive']['concentration'] = {}
    #     self.xls_coords['positive']['concentration']['usecols'] = 'B:M'
    #     self.xls_coords['positive']['concentration']['skiprows'] = 6
    #     self.xls_coords['positive']['concentration']['nrows'] = 1
    #     self.xls_coords['positive']['name'] = {}
    #     self.xls_coords['positive']['name']['usecols'] = 'N:N'
    #     self.xls_coords['positive']['name']['skiprows'] = 8
    #     self.xls_coords['positive']['name']['nrows'] = 8
    #     self.xls_coords['positive']['keyword'] = 'dC52'
    #
    #     self.xls_coords['negative'] = {}
    #     self.xls_coords['negative']['readout'] = {}
    #     self.xls_coords['negative']['readout']['usecols'] = 'B:M'
    #     self.xls_coords['negative']['readout']['skiprows'] = 8
    #     self.xls_coords['negative']['readout']['nrows'] = 8
    #     self.xls_coords['negative']['concentration'] = {}
    #     self.xls_coords['negative']['concentration']['usecols'] = 'B:M'
    #     self.xls_coords['negative']['concentration']['skiprows'] = 6
    #     self.xls_coords['negative']['concentration']['nrows'] = 1
    #     self.xls_coords['negative']['name'] = {}
    #     self.xls_coords['negative']['name']['usecols'] = 'N:N'
    #     self.xls_coords['negative']['name']['skiprows'] = 8
    #     self.xls_coords['negative']['name']['nrows'] = 8
    #     self.xls_coords['negative']['keyword'] = 'dC52'
    #
    #     self.xls_coords['blank'] = {}
    #     self.xls_coords['blank']['readout'] = {}
    #     self.xls_coords['blank']['readout']['usecols'] = 'B:M'
    #     self.xls_coords['blank']['readout']['skiprows'] = 8
    #     self.xls_coords['blank']['readout']['nrows'] = 8
    #     self.xls_coords['blank']['concentration'] = {}
    #     self.xls_coords['blank']['concentration']['usecols'] = 'B:M'
    #     self.xls_coords['blank']['concentration']['skiprows'] = 6
    #     self.xls_coords['blank']['concentration']['nrows'] = 1
    #     self.xls_coords['blank']['name'] = {}
    #     self.xls_coords['blank']['name']['usecols'] = 'N:N'
    #     self.xls_coords['blank']['name']['skiprows'] = 8
    #     self.xls_coords['blank']['name']['nrows'] = 8
    #     self.xls_coords['blank']['keyword'] = 'dC52'

    def read_format_fl(self, fl):
        """
        Reads the format file extracting the cell coordinates for the readouts to be processed, positives,
        negatives and blanks.
        :param fl:
        :return:
        """
        f = open(fl, 'r')
        for s in f:
            s = s.rstrip()
            if s == '' or s[0] == '#':
                continue
            wd_ls = s.split(' | ')
            wd_len = len(wd_ls)
            if wd_len > 4:
                exit('Fatal error, format line contains more that 4 elements:\n\t{}'.format(s))
            key0, key1, key2 = '', '', ''
            # print(s)
            for indx, item in enumerate(wd_ls):
                if indx == 0:
                    key0 = item
                    if key0 not in self.xls_coords.keys():
                        self.xls_coords[key0] = {}
                elif indx == 1:
                    key1 = item
                    if key1 not in self.xls_coords[key0].keys():
                        self.xls_coords[key0][key1] = {}
                elif indx == 2:
                    if indx == wd_len - 1:
                        self.xls_coords[key0][key1] = item
                    else:
                        key2 = item
                        if key2 not in self.xls_coords[key0][key1].keys():
                            self.xls_coords[key0][key1][key2] = {}
                elif indx == 3 and indx == wd_len - 1:
                    self.xls_coords[key0][key1][key2] = item
                else:
                    exit('Fatal error while reading the format')
        f.close()

    def out_from_user(self, targetdir):
        """
        Creates the Processing directory, the "data" subdirectory and a subdirectory for the excel file being
        processed
        :param targetdir: Output directory
        :return: outdir: Output directory corresponding to the processed excel file
        :return: excel_out: Output excel file with the raw, denoised and normalized data
        """
        # if not os.path.isdir(targetdir):
        #     exit('Output directory does not exist')
        if not os.path.isdir(targetdir):
            os.mkdir(targetdir)
        data_dir = targetdir + '/data'
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        outdir = data_dir + "/proc_" + re.sub(' ', '_', self.basename)  # directory of directory of file
        os.mkdir(outdir)
        excel_out = outdir + "/processed.xlsx"
        return outdir, excel_out

    def out_from_infl(self, fl):
        """
        sets up output directory, output excel and the basename
        :param fl: full path of the excel filename
        :return: outdir: Output directory corresponding to the processed excel file
        :return: excel_out: Output excel file with the raw, denoised and normalized data
        """
        outdir = os.path.dirname(fl) + "/proc_" + re.sub(' ', '_', self.basename)  # directory of directory of file
        # print("dirname is {}".format(outDir))
        if os.path.isdir(outdir):
            # os.rmdir(outdir)
            shutil.rmtree(outdir)
        os.mkdir(outdir)
        excel_out = outdir + "/processed.xlsx"
        return outdir, excel_out

    @staticmethod
    def get_basename(fl):
        """
        gets the basename of an input file (e.g. /home/User/Desktop/file.txt where the root is /home/User/Desktop/file)
        :param fl: full path
        :return: basename: root of the file
        """
        basename = os.path.basename(fl)
        basename = os.path.splitext(basename)[0]
        return basename

    def process_fl(self):
        """
        step by step processing of the excel file. 1) Read file; 2) denoise; 3) normalize;
        4) print and plot the corresponding dataframes
        :return:
        """
        df_exp, df_exp_group, ref_avmax, ds_min, blank_min, df_normal_group = self.initialize1()

        if self.denoise_option < 0 or self.denoise_option > 4:
            exit('wrong denoise option, it can only be 1, 2, 3, or 4')
        if self.normalize_option < 0 or self.normalize_option > 2:
            exit('wrong normalize options, it can only be 0, 1, 2')
        elif self.denoise_option < 3:
            # Read excel input and concatenate sheets into df_raw dataframe
            df_exp, df_exp_group, df_pos, df_neg, df_blank = self.read_xls_fl2()
            # df_exp.to_excel('readouts.xlsx')
            # Denoise and normalize
            if self.denoise_option == 0:
                self.df_denoised = df_exp
                df_denoised_group = df_exp_group
            if self.denoise_option == 1:
                ds_min, df_denoised_group = self.denoise_opt1(df_exp, df_exp_group)
            elif self.denoise_option == 2:
                blank_min, df_denoised_group = self.denoise_opt2(df_blank, df_exp, df_pos)
            if self.normalize_option == 1:
                # print('JODER')
                ref_avmax = 100;
                if self.denoise_option == 0 or self.denoise_option == 1:
                    ref_avmax = self.extract_ref_avmax(df_denoised_group)
                elif self.denoise_option == 2:
                    ref_avmax = df_pos.groupby('Specie').mean().select_dtypes(
                        include=['float64', 'int']).values.max() - blank_min
                self.normalize_df(self.df_denoised, ref_avmax)
        else:
            print('test 1')
            # Denoise and normalize within each sheet
            df_exp, df_exp_group, df_denoised_group = self.denoise_norm_opt3()
        if self.normalize_option != 0:
            df_normal_group = self.df_normal.groupby('Specie')
            df_normal_group.name = 'Normalized, grouped by specie'
        # Print
        # for key, item in df_normal_group:
        #     print(df_normal_group.get_group(key), "\n\n")
        self.print_dataframes(df_exp, df_exp_group, df_denoised_group, ds_min, blank_min, ref_avmax, df_normal_group)
        # self.writer.save()
        # self.workbook.close()
        # Plot raw data
        self.plot_dataframes(df_exp, df_exp_group, df_denoised_group, df_normal_group)
        # self.writer.save()
        self.writer.close()
        self.writer.handles = None


    def plot_dataframes(self, df_exp, df_exp_group, df_denoised_group, df_normal_group):
        """
        plots multiple dataframes
        :param df_exp: dataframe with the raw readouts
        :param df_exp_group: dataframe with the raw readouts grouped by specie
        :param df_denoised: dataframe with denoised readouts
        :param df_denoised_group: dataframe with denoised readouts grouped by specie
        :param df_normal_group: dataframe wieth normalized readouts grouped by specie
        :return:
        """
        color_lst = self.random_colors(df_exp_group)
        self.plot_df(self.outdir + '/raw.png', df_exp, df_exp_group, color_lst)
        self.plot_df(self.outdir + '/denoised.png', self.df_denoised, df_denoised_group, color_lst)
        if self.normalize_option > 0:
            self.plot_df(self.outdir + '/normalized.png', self.df_normal, df_normal_group, color_lst)
            self.plot_df_av(self.outdir + '/normalized_average.png', df_normal_group, color_lst)


    def print_dataframes(self, df_exp, df_exp_group, df_denoised_group, ds_min, blank_min, ref_avmax, df_normal_group):
        """
        Prints into an excel file raw and processed readouts
        :param df_exp:
        :param df_exp_group:
        :param df_denoised:
        :param df_denoised_group:
        :param ds_min:
        :param blank_min:
        :param ref_avmax:
        :param df_normal_group:
        :return:
        """
        self.print_df_concat('raw', df_exp, df_exp_group, "Nan")
        if self.denoise_option == 1:
            self.print_df_concat('denoised', self.df_denoised, df_denoised_group, ds_min)
        elif self.denoise_option == 2:
            self.print_df_concat('denoised', self.df_denoised, df_denoised_group,
                                 'Minimum average readout for blank: {}'.format(blank_min))
        elif self.denoise_option == 3:
            self.print_df_concat('denoised', self.df_denoised, df_denoised_group,
                                 'Denoising based on the minimum av readout for the blank within the sheet')

        if (self.denoise_option == 1 or self.denoise_option) == 2 and self.normalize_option == 1:
            self.print_df_concat('normalized', self.df_normal, df_normal_group,
                                 "Maximum average readout for normalization: {}".format(ref_avmax))
        elif self.normalize_option == 2:
            self.print_df_concat('normalized', self.df_normal, df_normal_group,
                                 "Nomalization based on maximum av readout within sheet")

    def denoise_opt2(self, df_blank, df_exp, df_pos):
        """
        Denoise and normalize readouts based on the union of blanks and positive (respectively) across sheets
        in the excel file. All readouts within the file are denoised and normalized using the same values
        Remove the background using the blank dataframe
        :param df_blank:
        :param df_exp:
        :param df_pos:
        :return:
        """
        blank_min = df_blank.groupby('Specie').mean().select_dtypes(include=['float64', 'int']).values.min()
        self.df_denoised = df_exp.iloc[:, :-1] - blank_min
        self.df_denoised['Specie'] = df_exp['Specie']
        # ref_avmax = df_pos.groupby('Specie').mean().select_dtypes(include=['float64', 'int']).values.max() - blank_min
        df_denoised_group = self.df_denoised.groupby('Specie')
        df_denoised_group.name = 'Denoised, grouped by specie'
        # self.normalize_df(self.df_denoised, ref_avmax)
        self.df_denoised.name = 'Denoised'
        return blank_min, df_denoised_group

    def denoise_opt1(self, df, df_group):
        """
        Denoise and normalize readouts based on the union of blanks and positive (respectively) across sheets
        in the excel file. All readouts within the file are denoised and normalized using the same values
        Remove background using the corresponding lowest average readout found for each sample
        :param df:
        :param df_group:
        :return:
        """
        ds_min = self.extract_min(df_group)
        # self.df_denoised = self.denoise_df(df, ds_min)
        self.denoise_df(df, ds_min)
        df_denoised_group = self.df_denoised.groupby('Specie')
        df_denoised_group.name = 'Denoised, grouped by specie'
        # ref_avmax = self.extract_ref_avmax(df_denoised_group)
        # self.normalize_df(self.df_denoised, ref_avmax)
        self.df_denoised.name = 'Denoised'
        return ds_min, df_denoised_group

    @staticmethod
    def initialize1():
        """
        Initializes dataframes and values
        :return:
        """
        # df_denoised = pd.DataFrame()
        df_exp = pd.DataFrame()
        df_exp_group = pd.DataFrame()
        df_normal_group = pd.DataFrame()
        ref_avmax = 'NaN'
        ds_min = pd.Series(dtype=float)
        blank_min = 0
        # return df_denoised, df_exp, df_exp_group, ref_avmax, ds_min, blank_min
        return df_exp, df_exp_group, ref_avmax, ds_min, blank_min, df_normal_group

    @staticmethod
    def plot_df_av(outfl, df_group, colorlst):
        """
        Plot the average and standard av readouts
        :param outfl: output file
        :param df_group: grouped dataframe
        :param colorlst: list of colors
        :return:
        """
        # denoised and normalized
        # fl_1 = re.sub('\.xlsx', '_' + key + '_average.png', in_fl)
        df_group_mean = df_group.mean()
        df_group_std = df_group.std()
        df_group_mean_t = df_group_mean.transpose()
        df_group_std_t = df_group_std.transpose()
        df_group_std_t.replace('', 0, inplace=True)
        df_group_std_t.to_csv('test.csv')
        # print(df_group_std_t)
        plt.figure()
        plt.rc('legend', fontsize=7)
        # dt_temp = df_group_std_t.dropna(how='all')
        # dt_temp.mask(dt_temp == 'NaN', inplace=True)
        # print(dt_temp)
        # ax = df_group_mean_t.plot(kind='line', yerr=df_group_std_t, subplots=False, capsize=3, color=colorlst)
        # warnings.filterwarnings("ignore")
        # I get warnings due to std dev in the grouped df where there is only one element
        # so I am just ignoring warnings when plotting
        ax = df_group_mean_t.plot(kind='line', yerr=df_group_std_t, subplots=False, capsize=3, color=colorlst)
        # warnings.resetwarnings()
        # warnings.filterwarnings("default")
        # ax = dt_temp.plot(kind='line', subplots=False, color=colorlst)
        plt.legend(fontsize=4)
        ax.set_xlabel("log10[rVSV]")
        ax.set_ylabel("Elisa signal (450nm)")
        plt.savefig(outfl, dpi=200)
        plt.close('all')

    @staticmethod
    def plot_df(outfl, df, df_group, colorlst):
        """
        Plots dataframes
        :param outfl: output file
        :param df: dataframe
        :param df_group: grouped dataframe
        :param colorlst: list of colors
        :return:
        """
        df_t = df.transpose()
        df_t.columns = df_t.loc['Specie']
        df_t = df_t.drop('Specie', axis=0)
        plt.figure()
        fig, ax = plt.subplots()
        ct = 0
        for name, group in df_group:
            # colr =np.random.rand(3,)
            colr = colorlst[ct]
            ax.plot(df_t[name], marker='o', linestyle='-', ms=3, label=name, color=colr)
            ct += 1
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=4)
        plt.xlabel("log10[rVSV]")
        plt.ylabel("Elisa signal (450nm)")
        plt.savefig(outfl, dpi=200)
        plt.close('all')

    def print_df_concat(self, sheet_name, df, df_group, s):
        """
        writes in a new excel spreadsheet the corresponding dataframes
        :param sheet_name: sheet name
        :param df: dataframe
        :param df_group: grouped dataframe
        :param s: either string or data.series
        :return:
        """
        worksheet = self.workbook.add_worksheet(sheet_name)
        self.writer.sheets[sheet_name] = worksheet
        row_ct = 0
        if isinstance(s, str) and s != "Nan":
            worksheet.write_string(row_ct, 0, s)
            row_ct += 1
        elif isinstance(s, pd.Series):
            worksheet.write_string(row_ct, 0, 'Minimum raw average for each specie')
            row_ct += 1
            s.to_excel(self.writer, sheet_name=sheet_name, startrow=row_ct, startcol=0)
            row_ct += s.shape[0] + 4
        worksheet.write_string(row_ct, 0, df.name)
        row_ct += 1
        df.to_excel(self.writer, sheet_name=sheet_name, startrow=row_ct, startcol=0)
        row_ct += df.shape[0] + 4
        av = df_group.mean()
        worksheet.write_string(row_ct, 0, 'Average')
        row_ct += 1
        av.to_excel(self.writer, sheet_name=sheet_name, startrow=row_ct, startcol=0)
        row_ct += av.shape[0] + 4
        std = df_group.std()
        worksheet.write_string(row_ct, 0, 'Standard deviation')
        row_ct += 1
        std.to_excel(self.writer, sheet_name=sheet_name, startrow=row_ct, startcol=0)

    @staticmethod
    def random_colors(df_group):
        """
        generates a list of random colors, one color per group in grouped dataframe
        :param df_group: grouped dataframe
        :return: color_lst: list of random colors
        """
        color_lst = []
        for name, group in df_group:
            colr = np.random.rand(3, )
            color_lst.append(colr)
        return color_lst

    def normalize_df(self, df, refmax):
        """
        Normalizes a dataframe based on a
        :param df: dataframe
        :param refmax: reference value for normalization
        :return: df_normal: normalized dataframe
        """
        self.df_normal = df.iloc[:, :-1] / refmax * 100
        # Zeroing out negative values
        self.df_normal[self.df_normal < 0] = 0
        self.df_normal['Specie'] = df['Specie']
        self.df_normal.name = 'Normalized'

    def extract_ref_avmax(self, df_group):
        """
        Idenfies the maximum average value (corresponding to a particular specie) in a dataframe
        :param df_group: dataframe grouped by specie
        :return: avmax: maximum average
        """
        # deno_ref = df_group.get_group(self.ref_sp)
        deno_ref = df_group.get_group(str(self.xls_coords['positive']['keyword']).upper())
        deno_ref_av = deno_ref.mean()
        # print(deno_dC52_av)
        avmax = deno_ref_av.max(axis=0)
        # print("\tMaximum average {}:\t{}".format(ref_sp, avmax))
        return avmax

    def denoise_df(self, df_raw, ds_min):
        """Substract a value extracted from a data series to the input dataframe
        :param df_raw: dataframe with raw elisa readouts
        :param ds_min: dataseries with the minimum average for each group/specie
        :return: df_denoised: dataframe with values substracted
        """
        # df_denoised = pd.DataFrame()
        for index in df_raw.index:
            # print(df_raw.iloc[index])
            sp = df_raw.iloc[index]['Specie']
            sp_min = ds_min.loc[sp]
            # print("Specie is {}, min is {}".format(sp, min))
            df = df_raw.iloc[index, :-1] - sp_min
            df['Specie'] = sp
            self.df_denoised = self.df_denoised.append(df, ignore_index=True)
        # print(df)
        # print(self.df_denoised)
        df_temp = self.df_denoised.drop('Specie', axis=1)
        df_temp[df_temp < 0] = 0
        df_temp['Specie'] = self.df_denoised['Specie']
        self.df_denoised = df_temp
        self.df_denoised.name = 'Denoised'

        # return df_denoised

    @staticmethod
    def extract_min(df_raw_group):
        """
        reads a grouped dataframe and returns the minimum average for each group
        :param df_raw_group: grouped dataframe
        :return: ds_min: data series describing the minimum average for each group (species)
        """
        ds_min = pd.Series(dtype='float64')
        group_mean = df_raw_group.mean()
        # print(group_mean)
        for name in group_mean.index:
            # print(group_mean.loc[name])
            ds = group_mean.loc[name]
            #    #print(group)
            # min = extract_refmin2(df)
            sp_min = ds.min()
            # print("Minimum for {} is {}".format(name, min))
            ds_min[name] = sp_min
        return ds_min

    def read_xls_fl(self):
        """
        Reads every sheet in the input file and returns a dataframe with the elisa readout
            :return: dataframe with the elisa readouts concatenated
        """
        sheet_ct = 0
        df_raw = pd.DataFrame()
        for sheet_nm in self.in_xls.sheet_names:
            worksheet = self.workbook.add_worksheet(sheet_nm)
            self.writer.sheets[sheet_nm] = worksheet
            print("\tReading sheet {}:\t{}".format(sheet_ct, sheet_nm))
            df0, df1, df2 = self.read_xls_sheet(sheet_ct)
            sheet_ct += 1
            # df3 = self.add_concentration(df1, df2)
            df4 = self.log_concentration(df2)
            self.print_parsed(worksheet, sheet_nm, df0, df2, df4)
            df_raw = df_raw.append(df4, ignore_index=True)
        print("\t...done")
        df_raw.name = 'Raw'
        return df_raw

    def read_xls_fl2(self):
        """
        Reads an excel file, sheet by sheet extracting readout data and classifying it in three dataframes: i)
        experimental data to be normalized and denoised; ii) positive readouts to be used while normalizing; iii)
        negative readouts (currently ignored) and iv) blank readouts to be used for denoising
        :return df_exp: experimental data to be normalized and denoised
        :return df_exp_group: experimental data to be normalized and denoised, sorted by specie
        :return df_pos: positive readouts to be used for normalizing
        :return df_neg: negative readouts
        :return df_blank: blank readouts to be used for denoising
        """
        sheet_ct = 0
        df_exp = pd.DataFrame()
        df_pos = pd.DataFrame()
        df_neg = pd.DataFrame()
        df_blank = pd.DataFrame()
        for sheet_nm in self.in_xls.sheet_names:
            worksheet = self.workbook.add_worksheet(sheet_nm)
            self.writer.sheets[sheet_nm] = worksheet
            print("\tReading sheet {}:\t{}".format(sheet_ct, sheet_nm))
            # Reading experimental values
            df_exp_conc = self.extract_df_in_xls_sheet(sheet_ct, 'experiment', 'concentration')
            df_exp_read = self.extract_df_in_xls_sheet(sheet_ct, 'experiment', 'readout')
            df_exp_name = self.extract_df_in_xls_sheet(sheet_ct, 'experiment', 'name')
            df_exp_conc.to_excel('concentration.xlsx')
            df_exp_read.to_excel('readout.xlsx')
            df_exp_name.to_excel('name.xlsx')
            keyword = self.xls_coords['experiment']['keyword']
            df_exp_comb = self.combine_df(df_exp_conc, df_exp_read, df_exp_name, keyword, 'not_equal',
                                          'Readouts to proces')
            # df_exp_comb.to_excel('joder3.xlsx')
            df_exp = df_exp.append(df_exp_comb, ignore_index=True)
            # Reading positive control
            df_pos_conc = self.extract_df_in_xls_sheet(sheet_ct, 'positive', 'concentration')
            df_pos_read = self.extract_df_in_xls_sheet(sheet_ct, 'positive', 'readout')
            df_pos_name = self.extract_df_in_xls_sheet(sheet_ct, 'positive', 'name')
            keyword = self.xls_coords['positive']['keyword']
            df_pos_comb = self.combine_df(df_pos_conc, df_pos_read, df_pos_name, keyword, 'equal',
                                          'Positive readouts (set the max)')
            df_pos = df_pos.append(df_pos_comb, ignore_index=True)
            # Reading negative control
            df_neg_conc = self.extract_df_in_xls_sheet(sheet_ct, 'negative', 'concentration')
            df_neg_read = self.extract_df_in_xls_sheet(sheet_ct, 'negative', 'readout')
            df_neg_name = self.extract_df_in_xls_sheet(sheet_ct, 'negative', 'name')
            keyword = self.xls_coords['negative']['keyword']
            df_neg_comb = self.combine_df(df_neg_conc, df_neg_read, df_neg_name, keyword, 'equal',
                                          'Negative readouts')
            df_neg = df_neg.append(df_neg_comb, ignore_index=True)
            # Reading blank control
            df_blank_conc = self.extract_df_in_xls_sheet(sheet_ct, 'blank', 'concentration')
            df_blank_read = self.extract_df_in_xls_sheet(sheet_ct, 'blank', 'readout')
            df_blank_name = self.extract_df_in_xls_sheet(sheet_ct, 'blank', 'name')
            keyword = self.xls_coords['blank']['keyword']
            df_blank_comb = self.combine_df(df_blank_conc, df_blank_read, df_blank_name, keyword, 'equal',
                                            'Blank readouts (set the background)')
            df_blank = df_blank.append(df_blank_comb, ignore_index=True)
            # Print
            self.print_parsed2(worksheet, sheet_nm, df_exp_comb, df_pos_comb, df_neg_comb, df_blank_comb)
            sheet_ct += 1
        print("\t...done")
        df_exp.name = 'Raw concatenated'
        df_exp_group = df_exp.groupby('Specie')
        df_exp_group.name = 'Raw, grouped by specie'
        # print('movinnng')
        return df_exp, df_exp_group, df_pos, df_neg, df_blank

    def denoise_norm_opt3(self):
        """
        Denoise and normalize readouts using only the blanks and positives in the corresponding excel sheet.
        Therefore, readouts in different sheets will be denoised and normalized using different blank and
        positive readouts
        :return:
        """
        sheet_ct = 0
        df_exp = pd.DataFrame()
        self.df_denoised = pd.DataFrame()
        df_normal = pd.DataFrame()
        for sheet_nm in self.in_xls.sheet_names:
            worksheet = self.workbook.add_worksheet(sheet_nm)
            self.writer.sheets[sheet_nm] = worksheet
            print("\tReading sheet {}:\t{}".format(sheet_ct, sheet_nm))
            # Reading experimental values
            df_exp_conc = self.extract_df_in_xls_sheet(sheet_ct, 'experiment', 'concentration')
            df_exp_read = self.extract_df_in_xls_sheet(sheet_ct, 'experiment', 'readout')
            df_exp_name = self.extract_df_in_xls_sheet(sheet_ct, 'experiment', 'name')
            keyword = self.xls_coords['experiment']['keyword']
            df_exp_comb = self.combine_df(df_exp_conc, df_exp_read, df_exp_name, keyword, 'not_equal',
                                          'Readouts_to_proces')
            df_exp = df_exp.append(df_exp_comb, ignore_index=True)
            # Reading positive control
            df_pos_conc = self.extract_df_in_xls_sheet(sheet_ct, 'positive', 'concentration')
            df_pos_read = self.extract_df_in_xls_sheet(sheet_ct, 'positive', 'readout')
            df_pos_name = self.extract_df_in_xls_sheet(sheet_ct, 'positive', 'name')
            keyword = self.xls_coords['positive']['keyword']
            df_pos_comb = self.combine_df(df_pos_conc, df_pos_read, df_pos_name, keyword, 'equal',
                                          'Positive readouts (set the max)')
            # df_pos = df_pos.append(df_pos_comb, ignore_index=True)
            # Reading negative control
            df_neg_conc = self.extract_df_in_xls_sheet(sheet_ct, 'negative', 'concentration')
            df_neg_read = self.extract_df_in_xls_sheet(sheet_ct, 'negative', 'readout')
            df_neg_name = self.extract_df_in_xls_sheet(sheet_ct, 'negative', 'name')
            keyword = self.xls_coords['negative']['keyword']
            df_neg_comb = self.combine_df(df_neg_conc, df_neg_read, df_neg_name, keyword, 'equal',
                                          'Negative readouts')
            # df_neg = df_neg.append(df_neg_comb, ignore_index=True)
            # Reading blank control
            df_blank_conc = self.extract_df_in_xls_sheet(sheet_ct, 'blank', 'concentration')
            df_blank_read = self.extract_df_in_xls_sheet(sheet_ct, 'blank', 'readout')
            df_blank_name = self.extract_df_in_xls_sheet(sheet_ct, 'blank', 'name')
            keyword = self.xls_coords['blank']['keyword']
            df_blank_comb = self.combine_df(df_blank_conc, df_blank_read, df_blank_name, keyword, 'equal',
                                            'Blank readouts (set the background)')
            # df_blank = df_blank.append(df_blank_comb, ignore_index=True)
            # Print
            self.print_parsed2(worksheet, sheet_nm, df_exp_comb, df_pos_comb, df_neg_comb, df_blank_comb)

            # denoise
            if self.denoise_option == 3:
                if len(df_blank_comb.index) > 0:
                    blank_min = df_blank_comb.groupby('Specie').mean().select_dtypes(include=['float64', 'int']).values.min()
                else:
                    print('Warning! I didnt get the blank, using 0 as blank')
                    blank_min = 0
            elif self.denoise_option == 4:
                blank_min = df_blank_comb.groupby('Specie').mean().select_dtypes(include=['float64', 'int']).values.max()
            df_denoised_temp = df_exp_comb.iloc[:, :-1] - blank_min
            df_denoised_temp[df_denoised_temp < 0] = 0
            df_denoised_temp['Specie'] = df_exp_comb['Specie']
            self.df_denoised = self.df_denoised.append(df_denoised_temp, ignore_index=True)

            # normalize
            if self.normalize_option == 2:
                ref_avmax = df_pos_comb.groupby('Specie').mean().select_dtypes(
                    include=['float64', 'int']).values.max() - blank_min
                df_normal_temp = df_denoised_temp.iloc[:, :-1] / ref_avmax * 100
                df_normal_temp[df_normal_temp < 0] = 0
                df_normal_temp['Specie'] = df_exp_comb['Specie']
                df_normal = df_normal.append(df_normal_temp, ignore_index=True)

            sheet_ct += 1
        print("\t...done")
        df_exp.name = 'Raw concatenated'
        df_exp_group = df_exp.groupby('Specie')
        df_exp_group.name = 'Raw, grouped by specie'
        self.df_denoised.name = 'Denoised'
        df_denoised_group = self.df_denoised.groupby('Specie')
        df_denoised_group.name = 'Denoised, grouped by specie'
        if self.normalize_option == 2:
            df_normal.name = 'Normalized'
            self.df_normal = df_normal
        return df_exp, df_exp_group, df_denoised_group

    def combine_df(self, df_conc, df_read, df_name, kw, condition, name):
        """
        Combines multiple dataframes (describing readout values, concentration and sample names) into a single
        dataframe
        :param df_conc: dataframe with epitope concentration
        :param df_read: dataframe with readout values
        :param df_name: dataframe with specie names
        :param kw: specie name used as reference (e.g. DC52 for blank)
        :param condition: either 'equal; or 'not_equal' this is used to include or exclude the specie
                            name described with kw
        :param name: name for the dataframe to be returned
        :return:
        """
        # Renaming column names starting from 0 in dataframe with readouts
        # df_read.columns = range(df_read.shape[1])
        # Adding dataframe with names to datagrame with readouts
        df_read = df_read.T.reset_index(drop=True).T
        # print(df_name)
        # print(df_conc)
        # print(df_read)
        df_name = df_name.replace('/', '-', regex=True)
        df_name = df_name.replace(' ', '', regex=True)
        # print(df_name)
        # df_name.to_excel('names.xlsx')
        df_read['Specie'] = df_name.astype(str)
        # print(df_read)
        df_read.iloc[:, -1] = df_read.iloc[:, -1].str.upper()
        # print(df_read)
        # print(0)
        # df_read.columns = [*df_read.columns[:-1], 'Specie']
        # print(df_read)
        # Add concentrations as column names
        # df_read.to_excel('joder4.xlsx')
        ct = 0
        first_item = 'NaN'
        # there is an issue renaming columns, I sort of fixed it by modifying all columns except to the
        # first column and then modifying the first columns outside a loop but I dont understand why this
        # is happenning
        for item in df_conc.iloc[0]:
            # print('\tcounter is', ct)
            # print('\tconcentration is', item)
            # print('renaming', ct, 'to', item)
            if ct == 0:
                first_item = float(item)
            else:
                df_read = df_read.rename(columns={int(ct): float(item)})
            # df_read.rename(columns = { df_read.columns[int(ct)]: item }, inplace=True)
            ct += 1
        # print('--------')
        df_read.rename(columns = { df_read.columns[0]: first_item}, inplace=True)
        # df_read.to_excel('joder5.xlsx')
        # sys.exit()
        # print(1)
        # print(df_read)
        # Converts concentrations to log
        # print('log transform, ', self.colhead_logt)
        if self.colhead_logt == True :
            # print('ja')
            # print('colhead is {}'.format(self.colhead_logt))
            for col in df_read.columns:
                # print("column name is {}".format(col))
                if col != 'Specie':
                    # print(col)
                    logitem = round(np.log10(col), 2)
                    # print('log calculation', col, logitem)
                    df_read = df_read.rename({col: logitem}, axis='columns')
                else:
                    df_read = df_read.rename({col: 'Specie'}, axis='columns')
        # print(2)
        # print(df_read)
        # Remove rows not matching the kewword
        if condition == 'not_equal':
            indexname = df_read[df_read['Specie'] == str(kw).upper()].index
        else:
            indexname = df_read[df_read['Specie'] != str(kw).upper()].index
        df_read.drop(indexname, inplace=True)
        df_read = self.remove_neg(df_read)
        df_read.name = name
        return df_read

    @staticmethod
    def remove_neg(df1):
        """
        Removes row in the dataframe where the Specie value starts with "NEG"
        :return:
        """
        df1 = df1.dropna(0,'all')
        # print(df1)
        filt = df1.Specie.str.contains("\*|^NEG|^NEGATIVE")
        df1 = df1[~filt]
        # print(df1)
        # emptying cells that are mark with an asterisk
        df2 = df1.replace(regex='^.+\*$|OVRFLW', value=np.nan)
        # print(df2)
        return(df2)

    def print_parsed(self, worksheet, nm, df_in0, df_in3, df_in4):
        """
        Prints in an excel spreadsheet the data parsed in the input excel file
        :param worksheet:
        :param nm: sheet name
        :param df_in0: dataframe with the procedure summary
        :param df_in3: dataframe with raw data and concentrations
        :param df_in4: dataframe with raw data and log10 concentration
        :return:
        """
        row_ct = 0
        worksheet.write_string(row_ct, 0, df_in0.name)
        row_ct += 1
        df_in0.to_excel(self.writer, sheet_name=nm, startrow=row_ct, startcol=0)
        row_ct += df_in0.shape[0] + 4
        worksheet.write_string(row_ct, 0, df_in3.name)
        row_ct += 1
        df_in3.to_excel(self.writer, sheet_name=nm, startrow=row_ct, startcol=0)
        row_ct += df_in3.shape[0] + 4
        worksheet.write_string(row_ct, 0, df_in4.name)
        row_ct += 1
        df_in4.to_excel(self.writer, sheet_name=nm, startrow=row_ct, startcol=0)

    def print_parsed2(self, worksheet, nm, df_exp, df_pos, df_neg, df_blank):
        """
        Prints in an excel spreadsheet the data parsed in the input excel file
        :param worksheet:
        :param nm: sheet name
        :param df_exp: dataframe with readouts for experimental values
        :param df_pos: dataframe with readouts for positives
        :param df_neg: dataframe with readouts for negatives
        :param df_blank: dataframe with readouts for blank
        :return:
        """
        row_ct = 0
        worksheet.write_string(row_ct, 0, df_exp.name)
        row_ct += 1
        df_exp.to_excel(self.writer, sheet_name=nm, startrow=row_ct, startcol=0)
        row_ct += df_exp.shape[0] + 4
        worksheet.write_string(row_ct, 0, df_pos.name)
        row_ct += 1
        df_pos.to_excel(self.writer, sheet_name=nm, startrow=row_ct, startcol=0)
        row_ct += df_pos.shape[0] + 4
        worksheet.write_string(row_ct, 0, df_neg.name)
        row_ct += 1
        df_neg.to_excel(self.writer, sheet_name=nm, startrow=row_ct, startcol=0)
        row_ct += df_neg.shape[0] + 4
        worksheet.write_string(row_ct, 0, df_blank.name)
        row_ct += 1
        df_blank.to_excel(self.writer, sheet_name=nm, startrow=row_ct, startcol=0)

    @staticmethod
    def log_concentration(df_in):
        """
        Converts the concentrations defining each column into the corresponding log10
        :param df_in: elisa readout with concentrations as column names
        :return:
            df_in: elisa readout with log10(concentration) as column names
        """
        for col in df_in.columns:
            # print("column name is {}".format(col))
            if col != 'Specie':
                logitem = round(np.log10(col), 2)
                df_in = df_in.rename({col: logitem}, axis='columns')
            else:
                df_in = df_in.rename({col: 'Specie'}, axis='columns')
        df_in.name = "Dataframe with log10concentration"
        return df_in

    @staticmethod
    def add_concentration(df_in1, df_in2):
        """
            Renames the columns in one dataframe based on the items in the remaining dataframe
            :param df_in1: dataframe with a single row containing concentration values
            :param df_in2: elisa readout
            :return:
                df_in2: elisa readout with concentrations as column names
            """
        ct = 1
        for item in df_in1.iloc[0]:
            # print("readout is {}\tlog:{}".format(item, logitem))
            df_in2 = df_in2.rename({ct: item}, axis='columns')
            ct += 1
        df_in2.name = "Dataframe with concentration"
        return df_in2

    def read_xls_sheet(self, ct):
        """
            Reads an excel sheet returning three dataframes. Attention, the excel data needs to be in a particular
             format
            :param ct: sheet page number
            :return:
                df0: dataframe with procedure summary
                df1: dataframe with concentrations
                df2: elisa readout
            """
        # Reading summary
        df_summary = pd.read_excel(self.in_xls, self.in_xls.sheet_names[ct], header=None, skiprows=1,
                                   nrows=4, usecols="A:B")
        df_summary.name = "Procedure Summary"
        df_exp_conc = pd.read_excel(self.in_xls, self.in_xls.sheet_names[ct], header=None, skiprows=6,
                                    nrows=1, usecols="B:M")

        # Reading experiment readouts
        skiprows = self.xls_coords['experiment']['readout']['skiprows']
        nrows = self.xls_coords['experiment']['readout']['nrows']
        usecols = self.xls_coords['experiment']['readout']['usecols']
        df_exp_read = pd.read_excel(self.in_xls, self.in_xls.sheet_names[ct], header=None, skiprows=skiprows,
                                    nrows=nrows, usecols=usecols)
        df_exp_name = pd.read_excel(self.in_xls, self.in_xls.sheet_names[ct], header=None, skiprows=8,
                                    nrows=8, usecols="N")
        df_exp_read['Specie'] = df_exp_name
        df_exp_read.iloc[:, -1] = df_exp_read.iloc[:, -1].str.upper()
        ct = 1
        for item in df_exp_conc.iloc[0]:
            # print("readout is {}\tlog:{}".format(item, logitem))
            df_exp_read = df_exp_read.rename({ct: item}, axis='columns')
            ct += 1
        df_exp_read.name = "Dataframe with concentration"
        # print(df_exp_read)
        return df_summary, df_exp_conc, df_exp_read

    def extract_df_in_xls_sheet(self, ct, key1, key2):
        """
        Returns a dataframe from a excelfile based on the input coordinates
        :param ct: sheet number
        :param key1: data to be extracted, either 'blank', 'positive', 'negative' or 'experiment'
        :param key2: data type, either 'concentration', 'readout', 'name'
        :return:
        """
        # print(key1, key2)
        skiprows = int(self.xls_coords[key1][key2]['skiprows'])
        nrows = int(self.xls_coords[key1][key2]['nrows'])
        usecols = self.xls_coords[key1][key2]['usecols']
        # print('nrows is ', nrows)
        # df = df_exp_read = pd.read_excel(self.in_xls, self.in_xls.sheet_names[ct], header=None,
        #                                  skiprows=skiprows, nrows=nrows, usecols=usecols)
        df = pd.read_excel(self.in_xls, self.in_xls.sheet_names[ct], header=None,
                           skiprows=skiprows, nrows=nrows, usecols=usecols)
        # Renaming column names starting from 0 in dataframe with readouts
        df.columns = range(df.shape[1])
        return df


if __name__ == '__main__':
    # format_fl = 'format/estefania_format.txt'
    # excel_fl = "/Users/gorkalasso/Documents/projects/ebola/Elisa_platform/Good Results" + \"/EBOV/Duplicate (MG)/2019-06-20 EBOV Validation ELISAs B51-B57.xlsx"
    # denoise_option = 1
    format_fl = '/Users/gorkalasso/Documents/projects/Elisa_platform/input_formats/covid19_a.txt'
    excel_fl = "/Users/gorkalasso/Documents/projects/Elisa_platform/covid19/JMD_excels/format_covid19_a/JMD110-121_RB_A-D_Ethan_glc.xlsx"
    denoise_option = 3
    elisa = ELISA(excel_fl, format_fl, '', denoise_option)
    elisa.process_fl()
