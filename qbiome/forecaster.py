import numpy as np

from quasinet import qnet

class Forecaster:
    """Forecast the data week by week by sequantially generating qnet predictions for the next timestamp and using the filled timestamp to update qnet predictions
    """

    def __init__(self, qnet_orchestrator):
        """Initialization

        Args:
            qnet_orchestrator (qbiome.QnetOrchestrator): an instance with a trained qnet model
        """
        self.qnet_orchestrator = qnet_orchestrator
        self.quantizer = qnet_orchestrator.quantizer

    def forecast_data(self,
                      data,
                      start_week,
                      end_week=None,
                      n_samples=100,
                      subject_id=None):
        """Forecast the data matrix from `start_week` to `end_week`

        Output format:

        |   subject_id | variable         |   week |    value |
        |-------------:|:-----------------|-------:|---------:|
        |            1 | Actinobacteriota |     27 | 0.36665  |
        |            1 | Bacteroidota     |     27 | 0.507248 |
        |            1 | Campilobacterota |     27 | 0.002032 |

        Args:
            data (numpy.ndarray): 2D array of label strings, produced by `self.get_qnet_inputs`
            start_week (int): start predicting from this week
            end_week (int, optional): end predicting after this week
            n_samples (int, optional): the number of times to sample from qnet predictions for one masked entry. Defaults to 100.
            subject_id (list[str], optional): subject id for each row of data. If None, we assume data has as many rows as quantized originally. This must be set for new patient data.

        Returns:
            pandas.DataFrame: see format above
        """
        if end_week is None:
            end_week = self.qnet_orchestrator.get_max_timestamp()
        forecasted_matrix = np.empty(data.shape)
        for idx, seq in enumerate(data):
            forecasted_seq = self.qnet_orchestrator.predict_sequentially_by_week(
                seq, start_week, end_week, n_samples=n_samples
            )
            forecasted_matrix[idx] = forecasted_seq

        if subject_id is None:
            df = self.quantizer.add_meta_to_matrix(forecasted_matrix)
        else:
            df = self.quantizer.add_meta_to_matrix(forecasted_matrix, add_subject_id=False)
            if len(subject_id) != df.index.size:
                raise('Subject_id list must match in length to number of rows in input data matrix in forecaster')
            df['subject_id']=subject_id
        # convert to plottable format
        plot_df = self.quantizer.melt_into_plot_format(df)
        return plot_df
