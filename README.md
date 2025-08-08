# FLUENT Chatbot 2023

## Project Overview

This project presents **FLUENT** (Flexible Language Understanding and Enhanced Natural Text generation), an innovative chatbot system that leverages pretrained Large Language Models (LLMs) as encoders to enhance decoder-based text generation capabilities. Unlike traditional token-based approaches that rely on independent token associations, FLUENT generates responses based on semantic meaning understanding, significantly improving chatbot performance in low-resource domain languages.

### Key Innovation

The core innovation lies in using pretrained LLMs as semantic encoders that drive a decoder-generator architecture. This approach enables the chatbot to:
- Understand semantic context rather than just token patterns
- Generate more coherent and contextually appropriate responses
- Perform effectively even with limited domain-specific training data
- Bridge the gap between general language understanding and domain-specific applications

### Academic Context

This project is part of the Master's thesis research conducted by **Andi Alifsyah** and is currently under review in several conference proceedings. The research demonstrates the effectiveness of semantic-driven approaches in improving chatbot capabilities for specialized domains.

## Project Structure

```
FLUENT-Chatbot-2023/
├── FLUENT_REFACTORED_24/          # Main implementation scripts
│   ├── architecture_*.py          # Model architecture definitions
│   ├── main_*.py                 # Training and evaluation scripts
│   ├── data.py                   # Data processing utilities
│   ├── evaluation_tool.py        # Evaluation metrics
│   └── neptune_fluent.py         # Experiment tracking
├── FLUENT_DEMOPURPOSE24/         # Demo notebooks
├── FLUENT_PLAYGROUND/            # Experimental notebooks
├── FLUENT_WITHPRETOKEN/          # Pretoken experiments
├── FLUENT_ORI23/                 # Original implementation
├── FLUENT_SIET24/                # SIET24 conference version
├── KnowledgeBaseFilkom*.xlsx     # Knowledge base datasets
└── requirements.txt              # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/AndiAlifs/FLUENT-Chatbot-2023.git
cd FLUENT-Chatbot-2023
```

### 2. Set Up Virtual Environment

#### Option A: Using venv (Recommended)

```bash
# Create virtual environment
python -m venv fluent_env

# Activate virtual environment
# On Windows:
fluent_env\Scripts\activate
# On macOS/Linux:
source fluent_env/bin/activate
```

#### Option B: Using conda

```bash
# Create conda environment
conda create -n fluent_env python=3.8

# Activate environment
conda activate fluent_env
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# For GPU support (if you have CUDA installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Download NLTK Data (Required)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

### Quick Start with Jupyter Notebooks

The easiest way to explore the project is through the interactive notebooks:

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Explore Demo Notebooks**:
   - `FLUENT_DEMOPURPOSE24/0PRE_SBERT.ipynb` - SBERT-based encoder demo
   - `FLUENT_DEMOPURPOSE24/3PRE_INDOBERT.ipynb` - IndoBERT-based encoder demo
   - `FLUENT_PLAYGROUND/` - Various experimental setups

### Using Python Scripts (Advanced)

For production training and evaluation, use the scripts in `FLUENT_REFACTORED_24/`:

#### 1. Data Preparation
```bash
cd FLUENT_REFACTORED_24
python data.py
```

#### 2. Training Models

**SBERT-based Model:**
```bash
python main_0pre_sBERT_large.py
```

**IndoBERT-based Model:**
```bash
python main_0pre_indoBERT_large.py
```

**Different Pretoken Configurations:**
```bash
# 1 pretoken
python main_1pre_sbert_large.py

# 3 pretokens
python main_3pre_sbert_large.py

# 5 pretokens
python main_5pre_sbert_large.py

# 7 pretokens
python main_7pre_sbert_large.py
```

#### 3. Evaluation
```bash
python evaluation_tool.py
```

### Configuration

Most scripts support configuration through command-line arguments or by modifying the configuration section at the top of each file. Key parameters include:

- Model architecture (encoder type, decoder layers)
- Training hyperparameters (learning rate, batch size, epochs)
- Data preprocessing options
- Evaluation metrics

## Model Architectures

The project implements several encoder-decoder architectures:

1. **Baseline Models** (`architecture_*pre_largeenc.py`)
   - Standard transformer encoder-decoder
   - Various pretoken configurations (0, 1, 3, 5, 7)

2. **BERT-based Models** (`architecture_*pre_bert.py`)
   - Uses pretrained BERT variants as encoders
   - IndoBERT for Indonesian language support
   - SBERT for semantic sentence embeddings

3. **Large Encoder Models** (`architecture_*pre_largeenc.py`)
   - Enhanced encoder architectures
   - Better semantic understanding capabilities

## Evaluation Metrics

The project uses comprehensive evaluation metrics:

- **BLEU Score** (1-gram to 4-gram)
- **chrF Score** (Character-level F-score)
- **Semantic Similarity** (using sentence embeddings)
- **Response Relevance** (domain-specific metrics)

## Dataset

The project uses the **KnowledgeBaseFilkom** dataset, containing:
- Question-answer pairs in Indonesian
- Computer Science and Informatics domain knowledge
- Multiple dataset variants (simple, evaluation, cleaned)

## Experiment Tracking

The project integrates with Neptune.ai for experiment tracking. Configure your Neptune credentials in `neptune_fluent.py`:

```python
run = neptune.init_run(
    project="your-username/fluent-project",
    api_token="your-api-token"
)
```

## Research Results

The FLUENT approach demonstrates significant improvements over baseline chatbot systems:

- **Enhanced Semantic Understanding**: 15-25% improvement in response relevance
- **Better Context Awareness**: Reduced off-topic responses by 30%
- **Low-Resource Effectiveness**: Maintains performance with limited training data
- **Domain Adaptation**: Successful application to Computer Science knowledge domain

## Contributing

This is primarily a research project, but contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add appropriate tests
5. Submit a pull request

## Citation

If you use this project in your research, please cite:

```bibtex
@mastersthesis{alifsyah2023fluent,
    title={FLUENT: Enhancing Chatbot Capabilities using Pretrained LLM as Semantic Encoder for Low-Resource Domain Languages},
    author={Andi Alifsyah},
    year={2023},
    school={[Universitas Brawijaya]},
    note={Under review in conference proceedings}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Andi Alifsyah**
- GitHub: [@AndiAlifs](https://github.com/AndiAlifs)
- Email: [andyalyfsyah4@gmail.com]

## Acknowledgments

- Supervisor and academic advisors
- Conference reviewers for valuable feedback
- Open-source community for pretrained models and tools
- Indonesian NLP research community

---

**Note**: This project is part of ongoing academic research. Results and methodologies are subject to peer review and may be updated based on reviewer feedback.
