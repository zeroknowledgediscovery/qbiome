import re
import string
import numpy as np
import pandas as pd

# helper functions for sorting
# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

class Quantizer():

    # TODO:
    # in the case where the conversion distortion from quantization-dequantization
    # is high, train models to minimize the distortion

    def __init__(self, num_levels=5):
        # use tuple for immutability
        self.num_levels = num_levels

        labels = tuple(string.ascii_uppercase[:num_levels])
        self.labels = {label: idx for idx, label in enumerate(labels)}
        self.variable_bin_map = {}
        self.column_names = [] # format {biome}_{week}

    def quantize_df(self, data):
        """
        must call quantize before calling any dequantize

        data: should be in melted format, produced by DataFormatter.load_data
        first two columns are sample_id and subject_id

        returns a pandas df
        """
        # some hacky intermediate format, so this probably shouldn't go into DataFormatter
        melted = pd.concat([
            data.subject_id,
            data.variable + '_' + data.week.astype(str),
            data.value
        ], axis=1).rename(columns={0: 'variable'})

        to_quantize = melted.pivot_table(
            index='subject_id', columns='variable')['value'].reset_index()
        self.column_names = to_quantize.columns[1:] # skip subject_id, only biome names
        # cache the subject_id column to add back to a dequantized matrix
        self.subject_id_column = to_quantize.subject_id

        quantized = pd.DataFrame() # return df
        for col in self.column_names:
            cut, bins = pd.cut(to_quantize[col], self.num_levels,
            labels=list(self.labels.keys()), retbins=True)
            quantized[col] = cut
            self.variable_bin_map[col] = bins

        # sort the columns by name in a natural order

        quantized = quantized.reindex(sorted(quantized.columns, key=natural_keys),
        axis=1)
        quantized.insert(0, 'subject_id', to_quantize.subject_id)
        return quantized

    def get_qnet_inputs(self, quantized_df):
        """
        use the quantized_df generated by quantize_df
        get the feature names and data matrix to feed into qnet
        returns (feature_names, matrix)
        """
        # skip subject_id column
        df = quantized_df.drop(columns='subject_id')
        return df.columns, df.values

    def dequantize_label(self, label, bin_arr):
        if label is np.nan or label == 'nan' or label not in self.labels:
            return np.nan
        low = self.labels[label]
        high = low + 1
        val = (bin_arr[low] + bin_arr[high]) / 2
        return val

    def dequantize_sequence(self, label_seq):
        """
        dequantize one row of the output matrix generated by the qnet

        letter_seq: numpy array

        returns a numpy array
        """
        numeric_seq = np.empty(label_seq.shape)
        for idx, label in enumerate(label_seq):
            col = self.column_names[idx]
            bin_arr = self.variable_bin_map[col]
            numeric_seq[idx] = self.dequantize_label(label, bin_arr)
        return numeric_seq

    def dequantize_to_df(self, label_matrix):
        """
        add back the subject_id column and the column names

        returns a pandas df
        """
        numeric_matrix = np.empty(label_matrix.shape)
        for idx, row in enumerate(label_matrix):
            numeric_matrix[idx] = self.dequantize_sequence(row)

        df = pd.DataFrame(numeric_matrix, columns=self.column_names)
        df = pd.concat([self.subject_id_column, df], axis=1)
        return df

    def melt_into_plot_format(self, data):
        """
        data: each column is {biome}_{week}
        """
        # pivot into plottable format
        melted = data.melt(id_vars='subject_id')
        # split variable names
        splitted = melted.variable.str.extract(r'([\D|\d]+)_(\d+)', expand=True)
        splitted.rename(columns={0: 'variable', 1: 'week'}, inplace=True)
        splitted.week = splitted.week.astype(int)
        plot_df = pd.concat([
            melted.subject_id, splitted, melted.value
        ], axis=1)
        return plot_df
