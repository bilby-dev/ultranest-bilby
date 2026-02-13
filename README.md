# ultranest-bilby

Plugin for using ultranest with bilby.

This plugin exposes the `ultranest` sampler via the `bilby.samplers` entry point.
Once installed, you can select it in `bilby.run_sampler` using `sampler='ultranest'`.

## Installation

The package can be install using pip

```
pip install ultranest-bilby
```

or conda

```
conda install conda-forge:ultranest-bilby
```

## Notes

- Interrupting when using `npool>1` does not work.

## Changes compared to original version

- `resume` no longer always set to `'overwrite'` (See commit: 48ab8169f4dab799d458d1602d082ae6f53ec286)
