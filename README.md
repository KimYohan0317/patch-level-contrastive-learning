# patch-level-contrastive-learning
This repository contains the official implementation of the **Patch-Level Contrastive Learning (PLCL)** model for time-series classification.

##  Abstract

Time-series classification is a crucial task in data analysis, widely applied across industries such as manufacturing, healthcare, and finance. Traditional approaches based on sequence models like LSTM and Transformer have achieved notable success but still struggle with inherent issues such as capturing long-term dependencies.

Recently, a growing body of research has focused on converting time-series data into images to overcome these limitations. One popular method, **Gramian Angular Fields (GAF)**, visually encodes temporal relationships, helping capture global trends often missed by sequence models. However, single-modality approaches that rely solely on one form of representation fail to comprehensively capture the diverse features of time-series data. Even existing multi-modal approaches often overlook fine-grained temporal patterns.

To address these challenges, we propose a **patch-level hybrid contrastive learning model** that fuses Transformers with Vision Transformers. By converting 1D time-series into GAF images and applying contrastive learning, the proposed model robustly learns local and global dependencies between timestamps and patches. Experimental results on the UCR time-series classification datasets demonstrate superior performance compared to prior methods, highlighting the effectiveness of our multi-modality integration approach. This work paves the way for broader applications of hybrid models in time-series classification tasks across various domains.

> **Note:** This work is based on an unpublished manuscript currently under submission.

## 📁 File Descriptions
- `dataset.py`: Defines a custom PyTorch Dataset class that loads time-series data and corresponding GASF images from a DataFrame. Returns a tuple of (time-series tensor, image, label) for each sample.
- `loss.py`: Implements a contrastive loss function for learning similarity between image and time-series patch embeddings, using temperature-scaled cross-entropy.
- `main.py`: Main training and evaluation script.
- `make_dataframe_new.py`: Loads UCR time-series data, applies label encoding, generates GASF and GADF images, saves them, and constructs metadata DataFrames with image paths and labels.
- `model.py`: Model architecture combining Transformer and Vision Transformer.
- `requirements.txt`: Required Python libraries.
- `utils.py`: Provides utility functions for calculating accuracy and plotting training metrics (e.g., loss over epochs). Plots are saved as PNG files.


## Requirements
The recommended requirements for PLCL are specified as follows:
- Python 3.9.12
- matplotlib==3.10.1
- numpy==2.2.5
- pandas==2.2.3
- Pillow==9.0.1
- Pillow==11.2.1
- pyts==0.13.0
- scikit_learn==1.4.1.post1
- torch==1.11.0
- torchvision==0.12.0
- tqdm==4.63.0
- transformers==4.40.2

You can install all dependencies by running:
```bash
pip install -r requirements.txt
```

## 📁 Dataset
Please download the UCR Time Series Classification datasets from the official site:

🔗 [UCR Time Series Classification Archive (2018)](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)

After downloading, organize the dataset directories according to your training pipeline.
