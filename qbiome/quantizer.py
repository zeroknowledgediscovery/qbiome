import re
import string
import pickle
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.ensemble import RandomForestRegressor

# helper functions for sorting
# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def _atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def _natural_keys(text):
    """
    alist.sort(key=_natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    """
    return [ _atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

class Quantizer:
    """Handles quantization and dequantization of data
    """

    def __init__(self, num_levels=5):
        """Initalization

        Args:
            num_levels (int, optional): Number of quantization levels. Defaults to 5.
        """
        # use tuple for immutability
        self.num_levels = num_levels
        """number of quantization levels"""

        labels = tuple(string.ascii_uppercase[:num_levels])
        self.labels = {label: idx for idx, label in enumerate(labels)}
        """ex. {A: 0, B: 1, ...}"""

        self.variable_bin_map = {}
        """key-value pairs {biome_name: quantization map}"""

        self.column_names = None
        """a list of columns in the format {biome}_{week}"""

        self.subject_id_column = None
        """cache this column to add back to the label matrix with `self.add_meta_to_matrix`"""

        self.random_forest_dict = {}
        """key-value pairs {biome_name: sklearn.ensemble.RandomForestRegressor}"""

    def save_quantizer_states(self, out_fname):
        """Save `self.column_names, self.subject_id_column, self.variable_bin_map, self.random_forest_dict`. Call this after calling `self.quantize_df`

        Args:
            out_fname (str): output file name
        """
        states = {
            'column_names': self.column_names,
            'subject_id_column': self.subject_id_column,
            'variable_bin_map': self.variable_bin_map,
            'random_forest_dict': self.random_forest_dict
        }
        with open(out_fname, 'wb') as f:
            pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_quantizer_states(self, in_fname):
        """Load in `self.column_names, self.variable_bin_map, self.random_forest_dict` from file

        Args:
            in_fname (str): input file name
        """
        with open(in_fname, 'rb') as f:
            states = pickle.load(f)
        self.column_names = states['column_names']
        self.subject_id_column = states['subject_id_column']
        self.variable_bin_map = states['variable_bin_map']
        self.random_forest_dict = states['random_forest_dict']

    def quantize_df(self, data):
        """This function must be called before calling any of the dequantization procedures. It populates `self.column_names, self.subject_id_column, self.variable_bin_map`

        input data format, produced by DataFormatter.load_data:

        | sample_id       |   subject_id | variable         |   week |    value |
        |:----------------|-------------:|:-----------------|-------:|---------:|
        | MBSMPL0020-6-10 |            1 | Actinobacteriota |     27 | 0.36665  |
        | MBSMPL0020-6-10 |            1 | Bacteroidota     |     27 | 0.507248 |
        | MBSMPL0020-6-10 |            1 | Campilobacterota |     27 | 0.002032 |
        | MBSMPL0020-6-10 |            1 | Desulfobacterota |     27 | 0.005058 |
        | MBSMPL0020-6-10 |            1 | Firmicutes       |     27 | 0.057767 |

        output data format:

        |   subject_id |   Acidobacteriota_35 | Actinobacteriota_1   |   Actinobacteriota_2 |
        |-------------:|---------------------:|:---------------------|---------------------:|
        |            1 |                  nan | A                    |                  nan |
        |           10 |                  nan | A                    |                  nan |
        |           11 |                  nan | A                    |                  nan |
        |           12 |                  nan | D                    |                  nan |
        |           14 |                  nan | A                    |                  nan |

        Args:
            data (pandas.DataFrame): see format above

        Returns:
            pandas.DataFrame: see format above
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
        """"""


        quantized = pd.DataFrame() # return df
        for col in self.column_names:
            cut, bins = pd.cut(to_quantize[col], self.num_levels,
            labels=list(self.labels.keys()), retbins=True)
            quantized[col] = cut
            self.variable_bin_map[col] = bins

        # sort the columns by name in a natural order

        quantized = quantized.reindex(sorted(quantized.columns, key=_natural_keys),
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
        matrix = df.astype(str).replace('nan', '').to_numpy(dtype=str)
        return df.columns, matrix

    def get_bin_array_of_index(self, idx):
        """
        get the pd.cut bin array corresponding to the sequence index
        """
        col = self.column_names[idx]
        bin_arr = self.variable_bin_map[col]
        return bin_arr

    def quantize_value(self, val, bin_arr):
        """
        quantize a numeric value to label
        should be the inverse of dequantize_label
        """
        label = pd.cut([val], bin_arr, labels=list(self.labels.keys()))[0]
        return label

    # procedures and helpers for dequantization follows

    def _fit_random_forest_one_biome(self, x, y):
        idx_old = np.arange(len(x))
        fx = interpolate.interp1d(idx_old, x, fill_value='extrapolate')
        fy = interpolate.interp1d(idx_old, y, fill_value='extrapolate')
        idx = np.arange(0, len(x), 0.01)
        X = fx(idx)[:, np.newaxis]
        Y = fy(idx)
        model = RandomForestRegressor()
        model.fit(X, Y)
        return model

    def compute_average_df(self, df):
        """
        df is in plot format, i.e., must have columns week, variable, value
        take avg wrt subject_id
        """
        avg = df[['variable', 'week', 'value']].groupby(
            by=['variable', 'week']).mean().reset_index()
        return avg

    def fit_random_forest(self, data, dequantized_data):
        """
        both data and dequantized_data are in plot format, i.e.,
        must have columns week, variable, value
        data: original data
        dequantized_data: produced by code like
        ```{python}
        df = self.quantizer.add_meta_to_matrix(matrix)
        dequantized_data = self.quantizer.melt_into_plot_format(df)
        ```

        writes in-place into self.random_forest_dict
        """
        if self.random_forest_dict:
            return

        # take avg of data and dequantized_data, grouped by week and biome
        # want to map dequantized to original, hence dequantized is input
        inputs = self.compute_average_df(dequantized_data)
        outputs = self.compute_average_df(data)

        for biome in inputs.variable.unique():
            x = inputs[inputs.variable == biome].value
            y = outputs[outputs.variable == biome].value
            model = self._fit_random_forest_one_biome(x, y)
            self.random_forest_dict[biome] = model

    def dequantize_label(self, label, bin_arr):
        if label is np.nan or label.lower() == 'nan' or label not in self.labels:
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
            bin_arr = self.get_bin_array_of_index(idx)
            numeric_seq[idx] = self.dequantize_label(label, bin_arr)
        return numeric_seq

    def dequantize_to_df(self, matrix):
        numeric_matrix = np.empty(matrix.shape)
        for idx, seq in enumerate(matrix):
            numeric_matrix[idx] = self.dequantize_sequence(seq)

        df = self.add_meta_to_matrix(numeric_matrix)
        return df

    def add_meta_to_matrix(self, matrix):
        """
        add back the subject_id column and the column names

        returns a pandas df
        """
        df = pd.DataFrame(matrix, columns=self.column_names)
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

    def apply_random_forest_regressor(self, data):
        """
        data: will be averaged and mapped to the average of original
        returns a df in plot format
        """
        if not self.random_forest_dict:
            raise Exception('No random forest models. First train with fit_random_forest')
        avg_data = self.compute_average_df(data)
        dataframes = []
        for biome in avg_data.variable.unique():
            x = avg_data[avg_data.variable == biome].value
            x = x.to_numpy()[:, np.newaxis]
            model = self.random_forest_dict[biome]
            pred = model.predict(x)
            df = pd.DataFrame({
                'variable': biome,
                'week': avg_data[avg_data.variable == biome].week,
                'value': pred
            })
            dataframes.append(df)
        ret = pd.concat(dataframes)
        return ret
