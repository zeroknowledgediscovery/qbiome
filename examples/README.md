# Examples

Notebooks in `local_development` are used for local development only, not the pip package.

Use `qbiome_example.ipynb, export_dotfiles.ipynb, hypothesis.ipynb`.

# Order of the pipeline

See the files in `local_development`

1. `data_formatter`
2. `quantizer`
3. `qnet_orchestrator`
4. `mask_checker`
5. `forecaster`
6. `export_dotfiles`
7. `hypothesis`

# Generate documentations

```
pdoc --html qbiome/ --output-dir docs --force
```

