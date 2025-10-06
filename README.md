# Automated Feature Synthesis for Relational Databases and Evaluation with TabPFN

This repository contains the working environment for my bachelor thesis **"Automated Feature Synthesis for Relational Databases and Evaluation with TabPFN"**.

The initial goal is to set up a local Python environment with [Featuretools](https://featuretools.alteryx.com/en/stable/) so that automated feature synthesis experiments can be run reproducibly. Later iterations will build on top of this foundation with research code and evaluation notebooks.

## Project structure

```
.
├── README.md                  # Project overview and usage instructions
├── requirements.txt           # Python dependencies to reproduce the environment
└── examples/
    └── featuretools_demo.py   # Minimal script showing Featuretools in action
```

## Prerequisites

- Python 3.9 or later (Featuretools currently supports Python 3.9 – 3.12).
- A virtual environment tool such as `venv`, `conda`, or `poetry`.

## Getting started

1. **Create and activate a virtual environment**. Below is an example using `conda`, followed by the original `venv` instructions.

   <details>
   <summary>Using conda</summary>

   ```bash
   conda create -n tabpfn-featuretools python=3.9
   conda activate tabpfn-featuretools
   ```

   </details>

   <details>
   <summary>Using the built-in <code>venv</code> module</summary>

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

   </details>

2. **Install the dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run the demo scripts** to verify that the tooling is installed and working:

   ```bash
   python examples/featuretools_demo.py
   python examples/tabpfn_demo.py
   python examples/featuretools_tabpfn_pipeline_demo.py
   python examples/relbench_small_pipeline.py  # downloads ~100MB on first run
   ```

The Featuretools script builds a small relational dataset, performs Deep Feature Synthesis (DFS), and prints the generated feature matrix along with a few synthesised features. The TabPFN script trains the probabilistic model on the classic Iris dataset and prints accuracy and a classification report. The combined pipeline example showcases how Featuretools-generated features can feed directly into a TabPFN classifier. If the scripts run successfully, the core tooling for the thesis is set up and ready for experimentation.

The RelBench demo downloads the small `rel-event` dataset (roughly 100 MB on the first run), engineers a handful of lightweight user-centric features, and evaluates a random forest baseline on the `user-attendance` regression task. You can customise the run via CLI flags, e.g. `--max-rows 2000` to keep the experiment light-weight or `--n-estimators 500` to strengthen the baseline.

## Next steps

- Expand the repository with data ingestion pipelines for relational datasets relevant to the thesis.
- Integrate the [TabPFN](https://github.com/automl/tabpfn) model for evaluating automatically generated features.
- Document experimental protocols and results as the thesis progresses.

## Troubleshooting

If installation fails, double-check that your Python version is supported. You can also try installing Featuretools directly to confirm the package is available:

```bash
pip install featuretools
```

For more detailed instructions and additional configuration options, consult the [official Featuretools documentation](https://featuretools.alteryx.com/en/stable/).
