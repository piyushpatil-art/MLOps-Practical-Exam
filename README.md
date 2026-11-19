# Wine Classification – Feature Store

## Overview
This module implements the Feature Store component of an end-to-end MLOps pipeline for Wine Classification using **Feast**. It manages and organizes dataset features for reuse during machine learning model training.

## Technologies Used
- Python
- Feast
- Pandas
- PyArrow
- Git & GitHub

## Features
- Wine dataset feature management
- Entity and Feature View definition
- Feature registration using Feast
- Local feature registry configuration
- Feature reuse for model training

## Configuration
The `feature_store.yaml` file configures the Feast project with a local registry and SQLite online store.

## Installation

```bash
pip install feast pandas pyarrow
```

## Usage

```bash
feast apply
feast feature-views list
feast entities list
```

## Project Structure

```text
feature_store/
├── feature_store.yaml
├── features.py
└── wine_features.parquet
```

## Objective
To provide organized and reusable features for the Wine Classification model as part of the MLOps workflow.

## Author
**Member 4 – Feature Store Implementation**
