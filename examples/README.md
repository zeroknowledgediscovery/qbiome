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
