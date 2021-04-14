import pandas as pd

class DataFormatter:
    """Parse raw data into usable format by the Quasinet
    """

    def __init__(self):
        pass

    def load_data(self, fpath_data, fpath_meta, taxon_name='Phylum',
    time_column_name='Age (days)', time_column_name_out='day',
    k_years=2, k_biomes=15):
        """Parse and join the data CSV and the metadata CSV

        Args:
            fpath_data (str): file path for the data CSV
            fpath_meta (str): file path for the metadata CSV
            taxon_name (str, optional): name of the taxon column exactly as in the data CSV.
            Defaults to 'Phylum'.
            time_column_name (str, optional): name of the timestamp column exactly as in the metadata CSV. Defaults to 'Age (days)'.
            time_column_name_out (str, optional): name of the timestamp column in the return data frame. Defaults to 'day'.
            k_years (int, optional): in the return data frame, we keep timestamps up to the number of years specified. Defaults to 2.
            k_biomes (int, optional): in the return data frame, we keep the k most abundant biomes. Defaults to 15.

        Returns:
            pandas.DataFrame: parsed, cleaned data frame
        """
        taxa_raw = pd.read_csv(fpath_data)
        meta_raw = pd.read_csv(fpath_meta)
        taxa_sum = self._sum_taxon(taxa_raw, taxon_name)
        meta = self._parse_meta(meta_raw, time_column_name, time_column_name_out)
        data = self._join_data_meta(taxa_sum, meta, time_column_name_out)

        # depending on the unit of the timestamp in the original data,
        # it may be necessary to cut out days beyond 2 or more years
        # and to convert days to weeks
        if k_years is not None:
            data = self._cut_after_k_years(data, k_years)
        data = self._convert_days_to_weeks(data)

        if k_biomes is not None:
            data = self._use_top_k_biomes(data, k_biomes)
        return data

    def pivot_into_column_format(self, data):
        """Pivot the input data frame from this format:

        | sample_id       |   subject_id | variable         |   week |    value |
        |:----------------|-------------:|:-----------------|-------:|---------:|
        | MBSMPL0020-6-10 |            1 | Actinobacteriota |     27 | 0.36665  |
        | MBSMPL0020-6-10 |            1 | Bacteroidota     |     27 | 0.507248 |
        | MBSMPL0020-6-10 |            1 | Campilobacterota |     27 | 0.002032 |
        | MBSMPL0020-6-10 |            1 | Desulfobacterota |     27 | 0.005058 |
        | MBSMPL0020-6-10 |            1 | Firmicutes       |     27 | 0.057767 |

        Into this format where each column is a biome:

        | sample_id         |   week |   Acidobacteriota |   Actinobacteriota |   Bacteroidota |
        |:------------------|-------:|------------------:|-------------------:|---------------:|
        | MBSMPL0020-6-421  |      1 |               nan |           0.011904 |       0.043808 |
        | MBSMPL0020-6-777  |      1 |               nan |           9.8e-05  |       0.000686 |
        | MBSMPL0020-6-1123 |      1 |               nan |           0.005603 |       0.201417 |
        | MBSMPL0020-6-1191 |      1 |               nan |           0.002578 |       0.368164 |
        | MBSMPL0020-6-263  |      1 |               nan |           0.004344 |       0.000381 |

        Args:
            data (pandas.DataFrame): see format above

        Returns:
            pandas.DataFrame: see format above
        """
        # keep sample_id in here for later cohort identification
        pivoted = data.pivot_table(
            index=['sample_id', 'week'], columns=['variable'])['value'].reset_index()
        pivoted.sort_values(by=['week'], inplace=True)
        pivoted.reset_index(drop=True, inplace=True)
        return pivoted

    def melt_into_plot_format(self, data):
        """Melt the data into a format `seaborn` can plot easily
        From format:

        | sample_id         |   week |   Acidobacteriota |   Actinobacteriota |   Bacteroidota |
        |:------------------|-------:|------------------:|-------------------:|---------------:|
        | MBSMPL0020-6-421  |      1 |               nan |           0.011904 |       0.043808 |
        | MBSMPL0020-6-777  |      1 |               nan |           9.8e-05  |       0.000686 |
        | MBSMPL0020-6-1123 |      1 |               nan |           0.005603 |       0.201417 |
        | MBSMPL0020-6-1191 |      1 |               nan |           0.002578 |       0.368164 |
        | MBSMPL0020-6-263  |      1 |               nan |           0.004344 |       0.000381 |

        Into format:

        | sample_id         |   week | variable        |   value |
        |:------------------|-------:|:----------------|--------:|
        | MBSMPL0020-6-421  |      1 | Acidobacteriota |     nan |
        | MBSMPL0020-6-777  |      1 | Acidobacteriota |     nan |
        | MBSMPL0020-6-1123 |      1 | Acidobacteriota |     nan |
        | MBSMPL0020-6-1191 |      1 | Acidobacteriota |     nan |
        | MBSMPL0020-6-263  |      1 | Acidobacteriota |     nan |

        Args:
            data (pandas.DataFrame): see format above

        Returns:
            pandas.DataFrame: see format above
        """
        melted = data.melt(id_vars=['sample_id', 'week'])
        return melted

    def _sum_taxon(self, taxa_raw, taxon_name):
        taxa = taxa_raw[['Sample ID', taxon_name, 'Relative Abundance']]
        taxa_sum = taxa.groupby(by=['Sample ID', taxon_name]).sum()
        taxa_sum.reset_index(inplace=True)
        taxa_sum.columns = ['sample_id', 'variable', 'value']
        print('There are {} unique biomes and {} unique samples'.format(
            len(taxa_sum.variable.unique()), len(taxa_sum.sample_id.unique())))
        return taxa_sum

    def _parse_meta(self, meta_raw, time_column_name, time_column_name_out):
        meta = meta_raw[['Sample ID', 'Property', 'Value']]

        meta_timestamp = meta[meta['Property'] == time_column_name].drop(columns='Property')
        meta_timestamp.columns = ['sample_id', time_column_name_out]

        meta_subject_id = meta[meta['Property'] == 'Subject ID'].drop(columns='Property')
        meta_subject_id.columns = ['sample_id', 'subject_id']

        meta = pd.merge(meta_timestamp, meta_subject_id, on='sample_id')
        return meta

    def _join_data_meta(self, data, meta, time_column_name):
        merged = pd.merge(data, meta, how='outer', on='sample_id')
        merged.columns = ['sample_id', 'variable', 'value', time_column_name, 'subject_id']
        merged.dropna(inplace=True)
        merged[time_column_name] = pd.to_numeric(merged[time_column_name],
        downcast='integer', errors='coerce').astype(int)
        # remove negative days
        merged = merged[merged[time_column_name] > 0]

        print('There are {} unique {}s'.format(
            len(merged[time_column_name].unique()), time_column_name))
        return merged

    def _cut_after_k_years(self, data, k_years):
        return data[data.day < 356 * k_years]

    def _convert_days_to_weeks(self, data):
        weeks = range(data.day.min() - 1, data.day.max() + 8, 7)
        print('There are {} unique weeks'.format(len(weeks)))
        data = pd.concat([
            data.sample_id,
            data.subject_id,
            data.variable,
            pd.cut(pd.Series(data.day), bins=weeks,
                        labels=range(1, len(weeks))),
            data.value
            ], axis=1)
        data.columns = ['sample_id', 'subject_id', 'variable', 'week', 'value']
        data.week = data.week.astype(int)
        return data

    def _use_top_k_biomes(self, data, k_biomes):
        """
        Everything except top k is labeled 'unclassified_Bacteria'
        """
        biome_measurement_counts = data.variable.value_counts()
        top_k = biome_measurement_counts.nlargest(k_biomes).index
        data.loc[~data.variable.isin(top_k), 'variable'] = 'unclassified_Bacteria'
        return data
