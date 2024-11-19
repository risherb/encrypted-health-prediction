---
title: Health Prediction On Encrypted Data Using Fully Homomorphic Encryption
emoji: 🩺😷
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
tags:
  - FHE
  - PPML
  - privacy
  - privacy preserving machine learning
  - image processing
  - homomorphic encryption
  - security
python_version: 3.10.6
---

# Healthcare prediction using FHE

## Running the application on your machine

From this directory, i.e., `health_prediction`, you can proceed with the following steps.

### Do once

First, create a virtual env and activate it:

<!--pytest-codeblocks:skip-->

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then, install required packages:

<!--pytest-codeblocks:skip-->

```bash
pip3 install pip --upgrade
pip3 install -U pip wheel setuptools --ignore-installed
pip3 install -r requirements.txt --ignore-installed
```

## Run the following steps each time you relaunch the application

In a terminal, run:

<!--pytest-codeblocks:skip-->

```bash
source .venv/bin/activate
python3 app.py
```

## Interacting with the application

Open the given URL link (search for a line like `Running on local URL:  http://127.0.0.1:8888/`).
