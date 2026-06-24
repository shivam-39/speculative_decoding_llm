# Speculative Decoding
** (Work-In-Progress)

This project focuses on optimizing speculative decoding for faster language model inference. 
The method uses a smaller, faster draft model to generate candidate tokens, which are then verified by a larger target model in a single forward pass.

## Project Layout

```text
speculative_decoding/
├── README.md
├── requirements.txt
└── src/
    ├── decoder.py
    └── generate.py
```

## Setup

Create a venv and install the dependencies:

```bash
conda create --name sp_dcode python=3.11
conda activate sp_dcode
pip install -r requirements.txt
```

## Run 

Speculative decoding:

```bash
python src/generate.py
```
