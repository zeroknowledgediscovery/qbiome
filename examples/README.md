# Order of the pipeline

1. `data_formatter`
2. `quantizer`
3. `qnet_orchestrator`
4. `mask_checker`
5. `forecaster`
6. `export_dotfiles`

# Generate documentations

```
pdoc --html qbiome/ --output-dir docs --force
```

# Bugs?

In the examples, there are 621 features. Sometimes for unknown non-deterministic reasons in the data formatter, there will be 622 features and the rest of the notebook will complain. Rerun the code (Restart & Clear Output) and most of the times you will get 621 features.
