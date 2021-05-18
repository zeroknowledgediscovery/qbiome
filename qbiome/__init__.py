"""
# Microbiome Analysis Powered By Recursive Quasi-species Networks:

Description:
    Uncovering rules of organization, competition, succession and exploitation

---

Installation:
   ```
      pip install qbiome 
   ```
   or
   ```
      pip3 install qbiome --user 
   ```

Notes:
   Depends on the `quasinet` package, which is 
   installed automatically. For visualization, the [graph-tool](https://graph-tool.skewed.de/) package is required, which is not installed automatically. Please install graph-tool separately for full functionality with directions from [here](https://graph-tool.skewed.de/download).

---

Quickstart:
   Data format
   
Examples:
   + Examples directory on github repo
   + [Medium article](https://medium.com/@ruolinzheng/18cb7da2baa5)
   

---

FAQ:
  + <u>`joblib errors`:</u> might require latest joblib version. In some python 3.8 installations, joblib 0.16.0 might be required. This is an open issue which is being addressed.
  + <u>`nan/Nan` appearing in quantized inputs:</u> `get_qnet_inputs` should not return any `nan/Nan`, which causes errors in downstream processing. The input to qnet inferrence should consist of letters `A,B,...` or the empty string for missing data.
  + <u>Infinite value error raised from `sklearn`:</u> This happens if `qbiome.forecaster` is not called prior to `Quantizer.apply_random_forest_regressor` with teh same strat week. Note that the forecaster step is expected to remove all missing data, without which the random forest regressor cannot be applied.

---

Contact:
  [Zero Knowledge Discovery](zed.uchicago.edu), The University of Chicago
  
  mailto: ishanu@uchicago.edu

"""
