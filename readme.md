# HallAgent4Rec: A Unified Framework for Reducing Hallucinations in LLM-Based Recommendation Agents

## 🚀 New Improved Workflow (No More 429 Errors!)

This implementation now uses a **two-phase approach** that eliminates 429 rate limit errors by pre-generating personality profiles separately from the main training process.

## 📋 Prerequisites

### 1. Install Dependencies
```bash
pip install numpy pandas scikit-learn sentence-transformers
pip install langchain langchain-google-genai tenacity
pip install matplotlib seaborn plotly streamlit
```

### 2. Download MovieLens 100K Dataset
```bash
wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
```

### 3. Set Google API Key
##### For Free Version
```bash
export GOOGLE_API_KEY="AIzaSyB_S8LYf2-YUD9ssMXqe9FzeWaqYEE90FI"
```
##### For Paid Version
```bash
export GOOGLE_API_KEY="AIzaSyB_S8LYf2-YUD9ssMXqe9FzeWaqYEE90FI"
```
## 🎯 Quick Start

### Option 1: Interactive Setup (Recommended)
```bash
python setup_personalities.py --interactive
```

### Option 2: Manual Step-by-Step

#### Step 1: Check Requirements
```bash
python setup_personalities.py --check-only
```

#### Step 2: Generate Personality Profiles (One-Time)
```bash
python generate_personalities.py --data_path ./ml-100k/ --output ./personalities.json
```
⏱️ **Time Required**: 30-60 minutes for MovieLens 100K (943 users)  
💰 **Estimated Cost**: ~$1-3 in API calls

#### Step 3: Run Main Analysis
```bash
python main.py --personalities_path ./personalities.json
```
⚡ **Time Required**: 5-10 minutes (no LLM calls during training!)

## 🔧 Detailed Usage

### Personality Generation Options

```bash
# Basic generation
python generate_personalities.py

# Custom settings
python generate_personalities.py \
    --data_path ./ml-100k/ \
    --output ./my_personalities.json \
    --batch_size 3 \
    --base_delay 5.0 \
    --batch_delay 20.0

# Resume interrupted generation
python generate_personalities.py --output ./personalities.json
# (automatically resumes from where it left off)

# Start fresh (ignore existing file)
python generate_personalities.py --no_resume
```

### Main Analysis Options

```bash
# Basic analysis with pre-generated personalities
python main.py --personalities_path ./personalities.json

# Skip online LLM phase (fastest)
python main.py --personalities_path ./personalities.json --skip_online

# Force personality regeneration during training
python main.py --force_generate

# Run with ablation study
python main.py --personalities_path ./personalities.json --ablation

# Different model configurations
python main.py --config tiny     # Fast testing
python main.py --config small    # Balanced
python main.py --config medium   # Best performance
```

## 🏗️ Architecture Overview

### Two-Phase Approach

**Phase 1: Personality Pre-Generation** (generate_personalities.py)
- Robust rate limiting (3s between calls, 15s between batches)
- Automatic retry with exponential backoff
- Resume capability for interrupted runs
- Progress tracking and validation
- Fallback responses for failed generations

**Phase 2: Main Training** (main.py)
- Load pre-generated personalities from JSON
- No LLM calls during offline training
- Optional online recommendation with LLM
- Fast and reliable execution

### Key Improvements

1. **Rate Limit Elimination**: No concurrent LLM calls during training
2. **Robustness**: Resume interrupted personality generation
3. **Efficiency**: Reuse personalities across experiments
4. **Debugging**: Separate personality validation and training
5. **Scalability**: Pre-generate once, experiment many times

## 📊 Monitoring and Analysis

### View Real-Time Progress
```bash
# During personality generation
tail -f personalities.json

# View dashboard after analysis
streamlit run analysis_dashboard.py
```

### Validate Personality Data
```python
from personality_generator import PersonalityVectorGenerator

generator = PersonalityVectorGenerator()
stats = generator.print_personality_stats("./personalities.json")
```

## 🛠️ Troubleshooting

### 429 Rate Limit Errors
❌ **Old Problem**: LLM calls during training loop caused rate limits  
✅ **Solution**: Use pre-generated personalities

```bash
# If you get 429 errors, check rate limiting settings
python generate_personalities.py --base_delay 5.0 --batch_delay 25.0
```

### API Key Issues
```bash
# Check API key
echo $GOOGLE_API_KEY

# Test connectivity
python -c "from utils import validate_api_setup; validate_api_setup()"
```

### Interrupted Generation
```bash
# Check progress
python setup_personalities.py --check-only

# Resume from where it left off
python generate_personalities.py --output ./personalities.json
```

### Memory Issues
```bash
# Use smaller configuration
python main.py --config tiny --personalities_path ./personalities.json
```

## 📁 File Structure

```
hallagent4rec/
├── generate_personalities.py    # 🆕 Pre-generate personality profiles
├── setup_personalities.py      # 🆕 Setup wizard and validation
├── main.py                     # 🔄 Updated main analysis (no LLM in training)
├── personality_generator.py    # 🔄 Updated to load from JSON
├── utils.py                    # 🔄 Enhanced rate limiting
├── personalities.json          # 🆕 Generated personality data
├── 
├── # Core framework files
├── hallagent4rec.py           # Main framework
├── collaborative_filtering.py  # Matrix factorization
├── attention_fusion.py        # Representation fusion
├── transfer_learner.py        # Transfer matrices
├── online_learner.py          # Online adaptation
├── hallucination_detector.py  # Hallucination mitigation
├── 
├── # Analysis and evaluation
├── evaluation.py              # Evaluation metrics
├── logger.py                  # Experiment logging
├── analysis_dashboard.py      # Results dashboard
├── config.py                  # Configuration
├── data_loader.py             # Data processing
└── ml-100k/                   # MovieLens dataset
```

## 🎯 Experiment Workflows

### Quick Testing
```bash
# 1. Generate personalities (once)
python generate_personalities.py --batch_size 3

# 2. Fast experiments
python main.py --config tiny --skip_online
```

### Full Research Run
```bash
# 1. Generate personalities with high quality
python generate_personalities.py --base_delay 3.0

# 2. Complete analysis with ablation
python main.py --config medium --ablation
```

### Production Analysis
```bash
# 1. Validate everything first
python setup_personalities.py --interactive

# 2. Generate with optimal settings
python generate_personalities.py --batch_size 5 --base_delay 3.0

# 3. Run full analysis
python main.py --personalities_path ./personalities.json
```

## 📈 Expected Results

After successful completion, you'll have:

- **personalities.json**: Pre-generated personality profiles (reusable)
- **experiments/**: Detailed experimental results and logs
- **Evaluation metrics**: Hit Rate, NDCG, MRR, RMSE, etc.
- **Visualizations**: Training curves, distributions, ablation results
- **Model files**: Trained HallAgent4Rec models

## 🆘 Support

If you encounter issues:

1. **Check setup**: `python setup_personalities.py --check-only`
2. **Validate API**: `python -c "from utils import validate_api_setup; validate_api_setup()"`
3. **Test small**: `python main.py --config tiny --skip_online`
4. **Check logs**: Look in `experiments/*/logs/experiment.log`

## 💡 Tips for Success

1. **Start Small**: Use `--config tiny` for initial testing
2. **Pre-generate Once**: Personality generation takes time but only needs to be done once
3. **Use Skip Online**: Add `--skip_online` to avoid LLM calls during testing
4. **Monitor Progress**: Keep an eye on API usage in the dashboard
5. **Resume Capability**: Interrupted personality generation can be resumed

---

**🎉 This new workflow eliminates the 429 error problem while making the framework more robust and efficient for research!**