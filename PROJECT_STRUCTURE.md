# Project Structure

This document details the file organization for the Amazon Product Recommendation System.

>  **[Back to Main README](README.md)** for setup and usage instructions.

---

##  Directory Tree

```
Project_Code/
│
├── notebooks/                           # Jupyter notebooks
│   ├── complete_preprocessing_pipeline.ipynb  #  Main preprocessing notebook
│   └── data_exploration.ipynb                 # Exploratory data analysis
│
├── data/                                # Data directory (created by notebook)
│   ├── raw/                            # Downloaded Amazon data
│   │   ├── Electronics_reviews.jsonl.gz      # Compressed reviews
│   │   ├── Electronics_reviews.jsonl         # Extracted reviews
│   │   ├── Electronics_metadata.jsonl.gz     # Compressed metadata
│   │   └── Electronics_metadata.jsonl        # Extracted metadata
│   │
│   ├── processed/                      # Processed datasets
│   │   ├── train.csv                   # Training set (CSV)
│   │   ├── validation.csv              # Validation set (CSV)
│   │   ├── test.csv                    # Test set (CSV)
│   │   ├── train.pkl                   # Training set (Pickle, faster)
│   │   ├── validation.pkl              # Validation set (Pickle)
│   │   ├── test.pkl                    # Test set (Pickle)
│   │   └── preprocessing_summary.txt   # Dataset statistics
│   │
│   └── mappings/                       # Feature encoders (for model training)
│       ├── user_id_encoder.pkl         # User ID → integer mapping
│       ├── item_id_encoder.pkl         # Item ID → integer mapping
│       ├── brand_encoder.pkl           # Brand → integer mapping
│       ├── category_encoder.pkl        # Category → integer mapping
│       └── vocab_sizes.pkl             # Vocabulary size summary
│
├── README.md                           #  Main documentation (you came from here!)
├── PROJECT_STRUCTURE.md                # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
└── 2023AC05492 (1).pdf                # Project reference (if any)
```

---

##  File Descriptions

### Notebooks

#### `complete_preprocessing_pipeline.ipynb` 
**Purpose**: Main preprocessing pipeline (all-in-one)

**Contains**:
- Setup and configuration
- Data download functions
- JSONL parsing
- Data cleaning and filtering
- Feature encoding
- Temporal splitting
- Verification and validation
- Data export
- Visualizations

**Dependencies**: All code is self-contained in the notebook

**Output**: Creates `data/processed/` and `data/mappings/` directories

#### `data_exploration.ipynb`
**Purpose**: Exploratory data analysis

**Requires**: Processed data from main notebook

**Includes**:
- User/item distribution analysis
- Temporal patterns
- Brand and category statistics
- Sparsity metrics
- Interaction distributions
- Visualizations

---

### Data Directories

#### `data/raw/`
**Purpose**: Store downloaded Amazon data

**Created by**: Notebook Section 2 (Data Download)

**Contents**:
- `.jsonl.gz` - Compressed downloads (original)
- `.jsonl` - Extracted JSON Lines files

**Size**: 1-5GB depending on category

**Note**: Git-ignored (too large)

#### `data/processed/`
**Purpose**: Store preprocessed train/val/test splits

**Created by**: Notebook Section 8 (Save Data)

**File formats**:
- `.csv` - Human-readable, slower to load
- `.pkl` - Pickle format, 10x faster loading

**Data schema**:
```
Columns (both CSV and Pickle):
- user_id              (original)
- user_id_encoded      (integer 0 to N-1)
- item_id              (original)
- item_id_encoded      (integer 0 to M-1)
- brand                (original)
- brand_encoded        (integer)
- category             (original)
- category_encoded     (integer)
- timestamp            (unix timestamp)
- feedback             (1 for all interactions)
- rating               (original rating, retained)
- title                (review title)
- text                 (review text)
```

**Size**: 100MB-1GB depending on filtering

**Note**: Git-ignored

#### `data/mappings/`
**Purpose**: Store feature encoders for model training

**Created by**: Notebook Section 5 (Feature Encoding)

**Files**:
```python
# Each encoder is a pickle file containing:
{
    'encoder': dict,           # original_value → encoded_integer
    'inverse_encoder': dict,   # encoded_integer → original_value
    'vocab_size': int          # number of unique values
}
```

**Usage example**:
```python
import pickle

# Load user encoder
with open('data/mappings/user_id_encoder.pkl', 'rb') as f:
    user_encoder = pickle.load(f)

# Get vocab size (for embedding layer)
n_users = user_encoder['vocab_size']

# Encode a user
user_encoded = user_encoder['encoder']['U12345']

# Decode back
user_original = user_encoder['inverse_encoder'][user_encoded]
```

**Note**: Essential for model training (keep in git with LFS or separate storage)

---

### Documentation

#### `README.md` 
**Purpose**: Main project documentation

**Sections**:
- Quick start guide
- Platform support (Mac/Linux/Windows)
- Configuration options
- Output files
- Troubleshooting
- Common tasks
- Next steps

#### `PROJECT_STRUCTURE.md` (this file)
**Purpose**: Detailed file organization

**Sections**:
- Directory tree
- File descriptions
- Usage examples
- Workflow diagram

#### `requirements.txt`
**Purpose**: Python package dependencies

**Install**: `pip install -r requirements.txt`

**Contents**:
```
torch>=2.1.0
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.8.0
seaborn>=0.13.0
tqdm>=4.66.0
jupyter>=1.0.0
```

---

##  Workflow

```
1. START
   │
   ├─→ Open: complete_preprocessing_pipeline.ipynb
   │
   ├─→ Edit: CONFIG dictionary (optional)
   │
   ├─→ Run: All cells (Cell → Run All)
   │
   ├─→ Creates:
   │   ├─→ data/raw/*.jsonl (downloads)
   │   ├─→ data/processed/*.csv & *.pkl (datasets)
   │   └─→ data/mappings/*.pkl (encoders)
   │
   ├─→ Open: data_exploration.ipynb (optional)
   │
   └─→ Ready for Phase 2: Model Implementation
```

---

##  Data Flow

```
Input (Amazon Data)
    ↓
[Download]
    ↓
data/raw/*.jsonl (10M+ reviews)
    ↓
[Load & Parse]
    ↓
DataFrame (reviews + metadata)
    ↓
[Filter]
    ↓
DataFrame (filtered: min 5 interactions)
    ↓
[Encode Features]
    ↓
DataFrame (with encoded columns) + Encoders
    │                                    ↓
    │                         data/mappings/*.pkl
    ↓
[Temporal Split]
    ↓
Train / Val / Test DataFrames
    ↓
[Save]
    ↓
data/processed/*.csv & *.pkl
    ↓
Ready for Model Training
```

---

##  Storage Requirements

| Dataset | Raw Data | Processed | Total |
|---------|----------|-----------|-------|
| Electronics | 2-3 GB | 500 MB | ~3.5 GB |
| Clothing | 2-3 GB | 400 MB | ~3.4 GB |
| Books | 8-10 GB | 1.5 GB | ~11 GB |

**Recommendation**: Have at least 10GB free disk space

---

##  Key Files for Model Training

When building models in Phase 2, you'll need:

1. **Datasets**:
   - `data/processed/train.pkl` (or .csv)
   - `data/processed/validation.pkl`
   - `data/processed/test.pkl`

2. **Encoders**:
   - `data/mappings/vocab_sizes.pkl` (for embedding dimensions)
   - `data/mappings/*_encoder.pkl` (for inference/decode predictions)

3. **Usage**:
   ```python
   import pandas as pd
   import pickle
   
   # Load training data
   train_df = pd.read_pickle('data/processed/train.pkl')
   
   # Load vocabulary sizes
   with open('data/mappings/vocab_sizes.pkl', 'rb') as f:
       vocab_sizes = pickle.load(f)
   
   n_users = vocab_sizes['user_id']
   n_items = vocab_sizes['item_id']
   ```

---

##  What's NOT Included

The following were removed during cleanup (now consolidated into notebook):

-  `src/` - Modular Python code
-  `config.yaml` - Configuration file
-  `run_preprocessing.py` - Standalone script
-  `setup_env.sh` - Environment setup script
-  `README_NOTEBOOK.md` - Notebook-specific docs
-  `MACOS_SUPPORT.md` - Mac-specific guide
-  `QUICKSTART_MAC.md` - Quick start guide

**All functionality is now in the main notebook!**

---

##  Naming Conventions

### Files
- Notebooks: `lowercase_with_underscores.ipynb`
- Data: `category_type.format` (e.g., `Electronics_reviews.jsonl`)
- Encoders: `feature_encoder.pkl` (e.g., `user_id_encoder.pkl`)

### Columns
- Original: `feature_name` (e.g., `user_id`, `item_id`)
- Encoded: `feature_name_encoded` (e.g., `user_id_encoded`)

---

##  Next Phase Structure

When you start Phase 2 (Model Implementation), you might add:

```
Project_Code/
├── notebooks/
│   ├── complete_preprocessing_pipeline.ipynb  (existing)
│   ├── data_exploration.ipynb                 (existing)
│   ├── model_sequential.ipynb                 (new - LSTM/Transformer)
│   ├── model_matrix.ipynb                     (new - Neural CF)
│   └── model_graph.ipynb                      (new - GNN)
│
├── models/                                     (new - saved models)
│   ├── lstm_best.pth
│   ├── ncf_best.pth
│   └── gnn_best.pth
│
└── results/                                    (new - evaluation results)
    ├── metrics.csv
    └── predictions.pkl
```

---

##  Checklist

Before starting model training, ensure you have:

- [ ] Completed preprocessing notebook
- [ ] Generated all files in `data/processed/`
- [ ] Generated all encoders in `data/mappings/`
- [ ] Reviewed data in exploration notebook
- [ ] Checked `preprocessing_summary.txt` for statistics
- [ ] Verified train/val/test split sizes
- [ ] Confirmed vocab sizes match expectations

---

**Ready to build models!** 

>  **[Back to Main README](README.md)** for next steps
