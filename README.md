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

1. **Create and activate a virtual environment** (shown with the built-in `venv` module):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

2. **Install the dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run the demo scripts** to verify that the tooling is installed and working:

   ```bash
   python examples/featuretools_demo.py
   python examples/tabpfn_demo.py
   ```

   The Featuretools script builds a small relational dataset, performs Deep Feature Synthesis (DFS), and prints the generated feature matrix along with a few synthesised features. The TabPFN script trains the probabilistic model on the classic Iris dataset and prints accuracy and a classification report. If both scripts run successfully, the core tooling for the thesis is set up and ready for experimentation.

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
