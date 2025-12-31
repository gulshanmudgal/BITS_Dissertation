# Amazon Product Recommendation System

A complete data preprocessing pipeline for building recommendation systems using Amazon product review data. Everything runs in a single Jupyter notebook - no complex setup required!

>  **[See Project Structure](PROJECT_STRUCTURE.md)** for detailed file organization.

---

##  Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Platform-specific notes:**
- **macOS (Apple Silicon)**: PyTorch automatically uses MPS for GPU acceleration
- **Linux/Windows (NVIDIA GPU)**: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- **CPU only**: Default installation works fine (slower but functional)

### 2. Run the Notebook

```bash
jupyter notebook notebooks/complete_preprocessing_pipeline.ipynb
```

### 3. Execute

In Jupyter: **Cell → Run All**

The notebook will:
1. Download Amazon product data
2. Clean and filter interactions  
3. Encode features to integers
4. Create train/val/test splits
5. Save processed datasets

**That's it!** 

---

##  What's Included

```
Project_Code/
├── notebooks/
│   ├── 01_data_pipeline_and_eda.ipynb     # Data preprocessing + EDA
│   ├── 02_matrix_factorization.ipynb      # MF baseline model
│   ├── 03_lstm_recommender.ipynb          # LSTM sequential model
│   ├── 04_bert4rec.ipynb                  # Transformer model
│   └── 05_lightgcn.ipynb                  # Graph Neural Network
├── data/                                   # Data directory (auto-created)
├── models/                                 # Saved model checkpoints
├── results/                                # Experiment results
├── README.md                               # This file
├── PROJECT_PLAN.md                         # Project tracking
└── requirements.txt                        # Dependencies
```

---

##  Main Notebook Overview

**`notebooks/complete_preprocessing_pipeline.ipynb`** contains 9 sections:

1. **Setup & Configuration** - Imports, device detection, settings
2. **Data Download** - Fetch from UCSD repository
3. **Load Data** - Parse JSONL to DataFrames
4. **Preprocessing** - Filter, clean, binarize
5. **Feature Encoding** - Map to integers for embeddings
6. **Temporal Splitting** - Leave-one-out train/val/test
7. **Verification** - Quality checks and plots
8. **Save Data** - Export CSV and pickle files
9. **Summary** - Final statistics

### Configuration

Edit the `CONFIG` dictionary at the top of the notebook:

```python
CONFIG = {
    'category': 'Electronics',         # Dataset: Electronics, Clothing_Shoes_and_Jewelry, Books
    'min_user_interactions': 5,        # Filter threshold
    'min_item_interactions': 5,
    'raw_data_dir': '../data/raw',
    'processed_data_dir': '../data/processed',
    'mappings_dir': '../data/mappings',
}
```

---

##  Platform Support

The notebook automatically detects your system and uses the best available device:

| Platform | GPU Support | Speed vs CPU | Setup Required |
|----------|-------------|--------------|----------------|
| **macOS (M1/M2/M3)** | MPS (Metal) | 5-10x | None  |
| **Linux/Windows (NVIDIA)** | CUDA | 10-20x | CUDA Toolkit |
| **CPU (Any)** | None | 1x | None  |

### macOS (Apple Silicon) Setup

 **Works out of the box!**

Verify MPS support:
```bash
# Check architecture
uname -m  # Should show "arm64"

# Verify MPS
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

**Memory Recommendations:**
- 8GB RAM: Small datasets (< 100K interactions)
- 16GB RAM: Medium datasets (100K-1M)  Recommended
- 32GB+ RAM: Large datasets (1M+)

### Linux/Windows (NVIDIA GPU) Setup

1. Install CUDA Toolkit 11.8+
2. Install PyTorch with CUDA:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```
3. Verify:
   ```bash
   nvcc --version
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

##  Output Files

After running the notebook, you'll find:

### Processed Datasets
```
data/processed/
├── train.csv              # Training interactions
├── validation.csv         # Validation interactions  
├── test.csv              # Test interactions
├── train.pkl             # Pickle (faster loading)
├── validation.pkl
├── test.pkl
└── preprocessing_summary.txt  # Statistics
```

### Feature Encoders
```
data/mappings/
├── user_id_encoder.pkl    # User ID → [0, N-1]
├── item_id_encoder.pkl    # Item ID → [0, M-1]
├── brand_encoder.pkl      # Brand → integers
├── category_encoder.pkl   # Category → integers
└── vocab_sizes.pkl        # Vocabulary counts
```

---

##  Common Tasks

### Process Full Dataset

```python
# In notebook, set max_lines=None
reviews_df = load_reviews(reviews_jsonl, max_lines=None)
metadata_df = load_metadata(metadata_jsonl, max_lines=None)
```

### Test with Sample Data

```python
# Process only 10K reviews for testing
reviews_df = load_reviews(reviews_jsonl, max_lines=10000)
```

### Change Dataset Category

```python
CONFIG = {
    'category': 'Books',  # Or 'Clothing_Shoes_and_Jewelry'
    # ...
}
```

### More Aggressive Filtering

```python
CONFIG = {
    'min_user_interactions': 10,  # Instead of 5
    'min_item_interactions': 10,
    # ...
}
```

### Load Processed Data (Later)

```python
import pandas as pd

# Fast loading with pickle
train_df = pd.read_pickle('../data/processed/train.pkl')

# Or use CSV
train_df = pd.read_csv('../data/processed/train.csv')
```

---

##  Troubleshooting

### Out of Memory

**Solution 1: Process Subset**
```python
reviews_df = load_reviews(reviews_jsonl, max_lines=50000)
```

**Solution 2: Aggressive Filtering**
```python
CONFIG['min_user_interactions'] = 10  # Higher threshold
```

**Solution 3: Restart Kernel**
- In Jupyter: Kernel → Restart & Clear Output

### Slow Download

- Large datasets take 10-30 minutes
- Files are cached (won't re-download)
- Progress bar shows status
- Can manually download and place in `data/raw/`

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### GPU Not Working

**macOS (Apple Silicon):**
```bash
# Must be arm64 architecture
uname -m

# Check MPS availability  
python -c "import torch; print(torch.backends.mps.is_available())"

# Reinstall if needed
pip install --upgrade torch
```

**Linux/Windows (NVIDIA):**
```bash
# Check CUDA
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Kernel Crashes

- Close other applications
- Reduce dataset size (use `max_lines`)
- Increase system swap/virtual memory
- Use more powerful machine

---

##  Dataset Information

### Source
Amazon Product Reviews Dataset from Hugging Face  
**URL**: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

### Available Categories

- **Electronics** (default) - ~20M reviews
- **Clothing, Shoes and Jewelry** - ~15M reviews  
- **Books** - ~50M reviews

### Data Format

**Reviews (input):**
- user_id, item_id (asin), rating, timestamp, title, text

**Metadata (input):**
- item_id (asin), title, brand, category

**Processed (output):**
- All features encoded to integers [0, N-1]
- Binary feedback (all interactions = 1)
- Temporal splits (train/val/test)

---

##  Next Steps

### 1. Explore the Data

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

Analyze:
- User/item distributions
- Temporal patterns
- Brand/category statistics
- Sparsity metrics

### 2. Phase 2: Build Models

Implement recommendation models:
- **Sequential**: LSTM, GRU, Transformer
- **Matrix**: Neural Collaborative Filtering, VAE
- **Graph**: Graph Neural Networks with PyTorch Geometric

### 3. Phase 3: Train & Evaluate

- Train on processed data
- Validate hyperparameters
- Test final performance
- Compare model architectures

### 4. Phase 4: Analysis

- Qualitative case studies
- Error analysis
- Performance comparison
- Recommendations visualization

---

##  Why Jupyter Notebook?

 **All-in-one** - Complete pipeline in single file  
 **Interactive** - See results as you go  
 **Self-contained** - No external code dependencies  
 **Easy to modify** - Edit and re-run cells  
 **Great for learning** - Read code + results together  
 **Reproducible** - Fixed random seeds  
 **Shareable** - Send single .ipynb file  

---

##  Dependencies

Core packages (see `requirements.txt`):

```
torch>=2.1.0          # Deep learning
pandas>=2.1.0         # Data manipulation
numpy>=1.24.0         # Numerical computing
scikit-learn>=1.3.0   # ML utilities
matplotlib>=3.8.0     # Plotting
seaborn>=0.13.0       # Statistical viz
tqdm>=4.66.0          # Progress bars
jupyter>=1.0.0        # Interactive notebooks
```

Install all:
```bash
pip install -r requirements.txt
```

---

##  Performance Tips

1. **Use GPU** - 5-20x faster than CPU
   - Mac: Automatic with M1/M2/M3
   - Linux/Win: Install CUDA

2. **Use Pickle** - Faster than CSV
   ```python
   df = pd.read_pickle('data/processed/train.pkl')
   ```

3. **Test First** - Use small sample before full run
   ```python
   max_lines=10000  # For testing
   ```

4. **Monitor Memory** - Use Activity Monitor (Mac) or Task Manager (Win)

5. **Close Apps** - Free RAM for processing

---

##  Educational Focus

This project teaches:
- Data preprocessing for RecSys
- Temporal splitting strategies
- Feature engineering
- Cross-platform ML pipelines
- Exploratory data analysis
- Jupyter best practices

Perfect for learning recommendation systems!

---

##  Citation

If using this dataset for research, please cite:
```
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

---

##  Contributing

This is an educational project. Feel free to:
- Modify configurations
- Try different datasets
- Add preprocessing steps
- Implement new models
- Share improvements

---

##  Features Summary

 Complete preprocessing pipeline  
 Cross-platform (Mac/Linux/Windows)  
 GPU acceleration (CUDA/MPS)  
 Interactive Jupyter notebook  
 Well-documented with examples  
 Integrity checks and validation  
 Multiple output formats (CSV/Pickle)  
 Temporal evaluation strategy  
 Feature encoding for embeddings  
 Reproducible with fixed seeds  

---

##  Getting Help

- Check **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** for file organization
- Review troubleshooting section above
- Read notebook markdown cells for details
- Check error messages - they're usually helpful!

---

##  Ready to Start?

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open notebook
jupyter notebook notebooks/complete_preprocessing_pipeline.ipynb

# 3. Run all cells (Cell → Run All)

# 4. Wait for completion (~10-30 min depending on dataset)

# 5. Find processed data in data/processed/
```

**Happy coding!** 

---

*Last updated: December 2025 | For educational use*
