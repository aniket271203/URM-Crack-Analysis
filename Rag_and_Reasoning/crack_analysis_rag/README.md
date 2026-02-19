# 🏗️ Structural Crack Analysis RAG System# 🏗️ Structural Crack Analysis RAG System



An AI-powered system for analyzing structural cracks using **Retrieval-Augmented Generation (RAG)** with Google's Gemini AI and a knowledge base of **2,600+ technical documents**.An AI-powered system for analyzing structural cracks using **Retrieval-Augmented Generation (RAG)** with Google's Gemini AI and a knowledge base of **2,600+ technical documents**.



## ✨ Key Features## ✨ Key Features



- **🤖 AI-Powered Analysis**: Uses Google Gemini for intelligent crack analysis- **🤖 AI-Powered Analysis**: Uses Google Gemini for intelligent crack analysis

- **📚 RAG System**: 2,626 document chunks from 15 authoritative sources (FEMA, research papers, standards)- **📚 RAG System**: 2,626 document chunks from 15 authoritative sources (FEMA, research papers, standards)

- **🔍 Enhanced Retrieval**: Query expansion, re-ranking, and smart context retrieval- **🔍 Enhanced Retrieval**: Query expansion, re-ranking, and smart context retrieval

- **🎯 Multi-step Analysis**: Location identification → Cause determination- **🎯 Multi-step Analysis**: Location identification → Cause determination

- **📊 5 Crack Types**: Diagonal, Vertical, Horizontal, Step, and X-shaped- **📊 5 Crack Types**: Diagonal, Vertical, Horizontal, Step, and X-shaped

- **📝 Detailed Reports**: Engineering-grade analysis with confidence levels- **📝 Detailed Reports**: Engineering-grade analysis with confidence levels



------



## 🚀 Quick Start## 🚀 Quick Start



### 1. Installation### 1. Installation



```bash```bash

# Navigate to project directory# Navigate to project directory

cd crack_analysis_ragcd crack_analysis_rag



# Install all dependencies# Install all dependencies

pip install -r requirements.txtpip install -r requirements.txt

``````



### 2. Setup API Key### 2. Setup API Key



Create a `.env` file in the project root:Create a `.env` file in the project root:



```bash```bash

# Create .env fileGEMINI_API_KEY=your_gemini_api_key_here

echo "GEMINI_API_KEY=your_gemini_api_key_here" > .envMODEL_NAME=gemini-2.5-flash-preview-05-20

```EMBEDDING_MODEL=all-MiniLM-L6-v2

```

Or manually create `.env` with:

```Get your API key from: https://makersuite.google.com/app/apikey

GEMINI_API_KEY=your_gemini_api_key_here

MODEL_NAME=gemini-2.5-flash-preview-05-20### 3. Process Documents (First Time Only)

EMBEDDING_MODEL=all-MiniLM-L6-v2

```If you haven't processed the documents yet, or want to add new PDFs:



**Get your API key:** https://makersuite.google.com/app/apikey```bash

# Place your PDF documents in ../RAG_Docs/ folder

### 3. Verify Setup# Then run:

python process_documents.py

```bash```

python -c "from src.main_rag import StructuralCrackRAG; print('✓ Setup successful!')"

```This will:

- Extract text from all PDFs

---- Create 2,600+ intelligent chunks

- Build vector embeddings

## 📖 Basic Usage- Save to FAISS database



### Analyze a Crack Image**Note:** This step is already done! You have 2,626 documents ready to use.



```python---

from src.main_rag import StructuralCrackRAG

from src.crack_types import CrackType## 📖 Usage



# Initialize the system### Basic Usage

rag = StructuralCrackRAG(data_dir="./data")

---

# Analyze a crack image with RAG

result = rag.analyze_crack(## 📖 Usage

    image_path="path/to/crack_image.jpg",

    crack_type=CrackType.DIAGONAL,  # Choose: DIAGONAL, VERTICAL, HORIZONTAL, STEP, X_SHAPED### Basic Usage

    use_rag=True  # Enable RAG for enhanced analysis

)```python

from src.main_rag import StructuralCrackRAG

# Check resultsfrom src.crack_types import CrackType

if result["success"]:

    print("✓ Analysis completed!")# Initialize the system

    print("\n📍 Location Analysis:")rag = StructuralCrackRAG(data_dir="./data")

    print(result["location_analysis"]["analysis"])

    print("\n🔍 Cause Analysis:")# Analyze a crack image with RAG

    print(result["cause_analysis"]["analysis"])result = rag.analyze_crack(

else:    image_path="path/to/crack_image.jpg",

    print(f"✗ Error: {result['error']}")    crack_type=CrackType.DIAGONAL,  # DIAGONAL, VERTICAL, HORIZONTAL, STEP, X_SHAPED

```    use_rag=True  # Enable RAG for better analysis

)

### Run Example Script

# Check results

```bashif result["success"]:

python example_usage.py    print("✓ Analysis completed!")

```    print("\nLocation Analysis:")

    print(result["location_analysis"]["analysis"])

### Available Crack Types    print("\nCause Analysis:")

    print(result["cause_analysis"]["analysis"])

| Crack Type | Use Case | Common Causes |else:

|-----------|----------|---------------|    print(f"✗ Error: {result['error']}")

| `CrackType.DIAGONAL` | Foundation issues | Settlement, shear stress |```

| `CrackType.VERTICAL` | Thermal cracks | Expansion, shrinkage |

| `CrackType.HORIZONTAL` | Beam failures | Flexural stress, overloading |### Command Line Usage

| `CrackType.STEP` | Masonry cracks | Differential movement |

| `CrackType.X_SHAPED` | Seismic damage | Shear failure, earthquakes |Use the provided example script:



---```bash

python example_usage.py

## 🎯 Complete Workflow```



### First Time Setup### Available Crack Types



```bash```python

# Step 1: Install dependenciesfrom src.crack_types import CrackType

pip install -r requirements.txt

CrackType.DIAGONAL    # Foundation settlement, shear stress

# Step 2: Create API key fileCrackType.VERTICAL    # Thermal expansion, shrinkage

echo "GEMINI_API_KEY=your_key_here" > .envCrackType.HORIZONTAL  # Flexural stress, beam overloading

CrackType.STEP        # Masonry differential movement

# Step 3: Verify setupCrackType.X_SHAPED    # Seismic damage, shear failure

python -c "from src.main_rag import StructuralCrackRAG; print('✓ Ready!')"```

```

---

### Analyze Cracks

## 🔧 Advanced Usage

```bash

# Option 1: Use example script### Add Custom Documents

python example_usage.py

```python

# Option 2: Create your own script# Add a single document

# See "Basic Usage" section aboverag.add_document(

```    content="Expert knowledge about diagonal cracks...",

    crack_type=CrackType.DIAGONAL,

### Test RAG System    title="Expert Analysis - Foundation Settlement",

    source="manual",

```bash    topics=["settlement", "foundation", "masonry"]

# Run comprehensive tests)

python test_enhanced_retrieval.py

```# Documents are automatically saved

```

### Add More Documents (Optional)

### Test Retrieval System

```bash

# Step 1: Place PDFs in ../RAG_Docs/ folder```bash

# Step 2: Process thempython test_enhanced_retrieval.py

python process_documents.py```

```

This will show you:

**Note:** Your system already has 2,626 documents ready! Only run this if you want to add more.- Query expansion in action

- Re-ranking results

---- Retrieved document quality

- System statistics

## 🔧 Advanced Usage

### Filter by Crack Type

### Custom Query for Better Context

```python

```pythonfrom src.enhanced_retrieval import EnhancedRetriever

# Customize the RAG query for better retrieval

result = rag.analyze_crack(retriever = EnhancedRetriever()

    image_path="crack.jpg",

    crack_type=CrackType.DIAGONAL,# Search with crack type filter

    use_rag=True,docs, metadata, scores = retriever.hybrid_search(

    rag_query="diagonal cracks in masonry walls due to foundation settlement"  # Custom query    document_manager=rag.document_manager,

)    query="causes of diagonal cracks in masonry",

```    k=10,  # Initial retrieval count

    final_k=5,  # Top results after re-ranking

### Add Your Own Documents    crack_type_filter=CrackType.DIAGONAL

)

```python

# Add expert knowledgefor doc, meta, score in zip(docs, metadata, scores):

rag.add_document(    print(f"Score: {score:.3f} | {meta['filename']}")

    content="Diagonal cracks in load-bearing walls often indicate differential settlement...",    print(f"{doc[:200]}...\n")

    crack_type=CrackType.DIAGONAL,```

    title="Expert Analysis - Foundation Settlement",

    source="manual",### Get Database Statistics

    topics=["settlement", "foundation", "masonry"]

)```python

```stats = rag.document_manager.get_stats()

print(f"Total Documents: {stats['total_documents']}")

### Search Documents Directlyprint(f"Crack Type Distribution: {stats['crack_type_distribution']}")

```

```python

from src.enhanced_retrieval import EnhancedRetriever---



retriever = EnhancedRetriever()## 📁 Project Structure



# Search with enhanced retrieval```

docs, metadata, scores = retriever.hybrid_search(crack_analysis_rag/

    document_manager=rag.document_manager,├── src/

    query="causes of diagonal cracks in masonry",│   ├── __init__.py

    k=10,  # Retrieve 10 initially│   ├── crack_types.py           # Crack type definitions

    final_k=5,  # Re-rank to top 5│   ├── document_manager.py      # Vector database management

    crack_type_filter=CrackType.DIAGONAL  # Optional filter│   ├── gemini_analyzer.py       # Gemini AI integration

)│   ├── enhanced_retrieval.py    # Smart search & re-ranking

│   └── main_rag.py              # Main RAG system

# Display results├── data/

for i, (doc, meta, score) in enumerate(zip(docs, metadata, scores), 1):│   ├── documents.json           # 2,626 document texts

    print(f"[{i}] Score: {score:.3f}")│   ├── metadata.json            # Document metadata

    print(f"    Source: {meta['filename']}")│   ├── faiss_index.pkl          # Vector embeddings

    print(f"    {doc[:200]}...\n")│   └── processed_chunks.json    # Processing details

```├── results/                     # Analysis results saved here

├── .env                         # API keys (create this)

### Get Statistics├── requirements.txt             # Python dependencies

├── process_documents.py         # PDF processing script

```python├── test_enhanced_retrieval.py   # Testing script

stats = rag.document_manager.get_stats()├── example_usage.py             # Usage examples

print(f"Total Documents: {stats['total_documents']}")└── README.md                    # This file

print(f"Distribution: {stats['crack_type_distribution']}")```

```

---

Expected output:

```## 🎯 Complete Workflow

Total Documents: 2626

Distribution: {'diagonal': 2597, 'horizontal': 26, 'vertical': 1, 'step': 1, 'x': 1}### Step 1: Setup (One-time)

```

```bash

---# Install dependencies

pip install -r requirements.txt

## 📁 Project Structure

# Create .env file with your API key

```echo "GEMINI_API_KEY=your_key_here" > .env

crack_analysis_rag/

├── src/# Verify installation

│   ├── crack_types.py           # Crack type definitionspython -c "from src.main_rag import StructuralCrackRAG; print('✓ Setup successful!')"

│   ├── document_manager.py      # Vector database (FAISS)```

│   ├── gemini_analyzer.py       # Gemini AI integration

│   ├── enhanced_retrieval.py    # Smart search & re-ranking### Step 2: Process Documents (If needed)

│   └── main_rag.py              # Main RAG orchestrator

├── data/```bash

│   ├── documents.json           # 2,626 document texts# Only run if you have new PDFs to add

│   ├── metadata.json            # Document metadatapython process_documents.py

│   └── faiss_index.pkl          # Vector embeddings```

├── results/                     # Analysis results saved here

├── .env                         # API keys (you create this)### Step 3: Analyze Cracks

├── requirements.txt             # Dependencies

├── process_documents.py         # PDF processor```bash

├── test_enhanced_retrieval.py   # Test script# Use the example script

├── example_usage.py             # Usage examplespython example_usage.py

└── README.md                    # This file

```# Or create your own script

python your_analysis_script.py

---```



## 📚 Document Sources### Step 4: Review Results



Your RAG system includes **2,626 chunks** from **15 authoritative sources**:Results are saved in the `results/` folder with timestamps:

- `{crack_type}_{timestamp}_report.txt` - Human-readable report

| Source | Chunks | Description |- `{crack_type}_{timestamp}.json` - Structured data

|--------|--------|-------------|

| FEMA P-154 Handbook | 816 | Seismic screening & assessment |---

| ASCE Guidelines | 910 | Structural evaluation standards |

| Materials Science Papers | 321 | Research on crack mechanisms |## 🧪 Testing & Validation

| Masonry Guides | 171 | Brick & block maintenance |

| Concrete Repair Standards | 36 | Repair techniques & methods |### Test the RAG System

| GSA Monitoring Procedures | 9 | Government monitoring protocols |

| Technical Research Papers | 363 | Additional academic sources |```bash

# Comprehensive test of retrieval system

**Topics Covered:**python test_enhanced_retrieval.py

- ✅ Inspection & assessment methods```

- ✅ Repair & remediation techniques  

- ✅ Crack causes & failure mechanismsOutput includes:

- ✅ Monitoring & measurement procedures- 6 test queries across different crack types

- ✅ Structural analysis & load evaluation- Query expansion demonstration

- ✅ Seismic considerations & earthquake damage- Re-ranking effectiveness

- ✅ Material properties (masonry, concrete)- Formatted context examples



---### Verify Document Database



## 🧪 Testing & Validation```python

from src.main_rag import StructuralCrackRAG

### Run Comprehensive Tests

rag = StructuralCrackRAG(data_dir="./data")

```bashstats = rag.document_manager.get_stats()

python test_enhanced_retrieval.py

```print(f"✓ {stats['total_documents']} documents loaded")

print(f"✓ Index size: {stats['index_size']}")

This tests:print(f"✓ Distribution: {stats['crack_type_distribution']}")

- ✅ Query expansion (synonyms, domain terms)```

- ✅ Re-ranking algorithm (multi-factor scoring)

- ✅ Crack type filteringExpected output:

- ✅ Context formatting```

- ✅ Retrieval quality✓ 2626 documents loaded

✓ Index size: 2626

### Quick Verification✓ Distribution: {'diagonal': 2597, 'horizontal': 26, 'vertical': 1, ...}

```

```bash

# Check if documents are loaded---

python -c "from src.main_rag import StructuralCrackRAG; print(StructuralCrackRAG().document_manager.get_stats())"

```## 📚 Document Sources



Should output:Your RAG system includes 2,626 chunks from these authoritative sources:

```

{'total_documents': 2626, 'crack_type_distribution': {...}, 'index_size': 2626}1. **FEMA P-154** - Seismic screening handbook (816 chunks)

```2. **ASCE Guidelines** - Structural assessment (910 chunks)

3. **Materials Science Research** - Academic papers (321 chunks)

---4. **Masonry Guides** - Brick & block maintenance (171 chunks)

5. **GSA Monitoring** - Government procedures (9 chunks)

## ⚙️ Configuration6. **Repair Standards** - Concrete & masonry repair (36 chunks)

7. **Technical Manuals** - 8 additional research papers

### Environment Variables

**Topics Covered:**

Edit `.env` file:- ✅ Inspection methods

- ✅ Repair techniques

```bash- ✅ Crack causes & mechanisms

# Required- ✅ Monitoring procedures

GEMINI_API_KEY=your_api_key_here- ✅ Structural analysis

- ✅ Seismic considerations

# Optional (with defaults)- ✅ Material properties

MODEL_NAME=gemini-2.5-flash-preview-05-20

EMBEDDING_MODEL=all-MiniLM-L6-v2---

```

## ⚙️ Configuration

### Adjust Retrieval Count

### Environment Variables (.env)

Edit `src/main_rag.py` around line 175:

```bash

```python# Required

docs, metadata, scores = self.enhanced_retriever.hybrid_search(GEMINI_API_KEY=your_gemini_api_key

    document_manager=self.document_manager,

    query=rag_query,# Optional (defaults shown)

    k=10,        # Change this: initial retrieval countMODEL_NAME=gemini-2.5-flash-preview-05-20

    final_k=5,   # Change this: final re-ranked resultsEMBEDDING_MODEL=all-MiniLM-L6-v2

    crack_type_filter=crack_type```

)

```### Adjust Retrieval Settings



### Adjust Re-ranking WeightsEdit `src/main_rag.py` line ~175:



Edit `src/enhanced_retrieval.py` lines 99-130 to modify boost factors:```python

docs, metadata, scores = self.enhanced_retriever.hybrid_search(

```python    document_manager=self.document_manager,

# Boost for exact query match    query=rag_query,

if query.lower() in content_lower:    k=10,        # Change: Initial retrieval count

    score *= 1.3  # Change this multiplier    final_k=5,   # Change: Final re-ranked results

    crack_type_filter=crack_type

# Boost for crack type match)

if query_crack_type.value in result_crack_types:```

    score *= 1.4  # Change this multiplier

```### Adjust Chunking Settings



---Edit `process_documents.py` lines 29-30:



## 🐛 Troubleshooting```python

processor = DocumentProcessor(

### "Module not found" or Import Error    chunk_size=1000,     # Change: Characters per chunk

    chunk_overlap=200    # Change: Overlap between chunks

```bash)

# Reinstall dependencies```

pip install -r requirements.txt --upgrade

---

# Verify you're in the correct directory

pwd  # Should show: .../crack_analysis_rag## 🐛 Troubleshooting

```

### Issue: "Import Error" or Module Not Found

### API Key Issues

```bash

```bash# Make sure you're in the right directory

# Check if .env existscd crack_analysis_rag

cat .env

# Reinstall dependencies

# Should show: GEMINI_API_KEY=your_keypip install -r requirements.txt --upgrade

```

# If missing, create it:

echo "GEMINI_API_KEY=your_key_here" > .env### Issue: "API Key Error"

```

```bash

### No Documents Loaded# Verify .env file exists and has correct key

cat .env

```bash

# Check if data files exist# Should show: GEMINI_API_KEY=your_actual_key_here

ls -lh data/```



# Should show: documents.json, metadata.json, faiss_index.pkl### Issue: "No documents loaded"



# If missing, run:```bash

python process_documents.py# Check if data files exist

```ls -lh data/



### Poor Retrieval Quality# Should show: documents.json, metadata.json, faiss_index.pkl



1. Adjust chunk size in `process_documents.py` (lines 29-30)# If missing, run:

2. Modify boost factors in `src/enhanced_retrieval.py` (lines 99-130)python process_documents.py

3. Add more domain-specific query expansions (lines 20-47)```



### Memory Errors During Processing### Issue: Poor retrieval quality



Edit `process_documents.py` line 429:```python

# Adjust re-ranking boost factors in src/enhanced_retrieval.py

```python# Lines 99-130 contain boost multipliers

batch_size = 25  # Reduce from 50```

```

### Issue: Memory errors

---

```python

## 🚀 Performance Optimization# Reduce batch size in process_documents.py line 429:

batch_size = 25  # Instead of 50

### Faster Processing (Lower Quality)```



```python---

rag = StructuralCrackRAG(

    embedding_model="all-MiniLM-L6-v2"  # Fast, smaller model## 🚀 Performance Tips

)

```### For Faster Processing



### Better Accuracy (Slower)```python

# Use smaller embedding model

```pythonrag = StructuralCrackRAG(

rag = StructuralCrackRAG(    embedding_model="all-MiniLM-L6-v2"  # Fast

    embedding_model="all-mpnet-base-v2"  # Better quality, larger model)

)```

```

### For Better Accuracy

### GPU Acceleration

```python

```bash# Use larger embedding model

# Uninstall CPU versionrag = StructuralCrackRAG(

pip uninstall faiss-cpu    embedding_model="all-mpnet-base-v2"  # Better but slower

)

# Install GPU version (requires CUDA)```

pip install faiss-gpu

```### For GPU Acceleration



---```bash

# Uninstall CPU version

## 📊 System Capabilitiespip uninstall faiss-cpu



| Feature | Status | Details |# Install GPU version (requires CUDA)

|---------|--------|---------|pip install faiss-gpu

| PDF Processing | ✅ | Automatic text extraction |```

| Vector Database | ✅ | FAISS with 2,626 documents |

| Query Expansion | ✅ | 100+ domain synonyms |---

| Smart Re-ranking | ✅ | 6 relevance factors |

| Image Analysis | ✅ | Gemini Vision AI |## 📊 System Capabilities

| 5 Crack Types | ✅ | Complete coverage |

| Export Reports | ✅ | JSON + TXT formats || Feature | Status | Details |

| Metadata Tracking | ✅ | Topics, sources, types ||---------|--------|---------|

| Document Processing | ✅ | 2,626 chunks ready |

---| Vector Search | ✅ | FAISS-based |

| Query Expansion | ✅ | Domain-specific synonyms |

## 🔍 How It Works| Re-ranking | ✅ | Multi-factor scoring |

| Crack Types | ✅ | 5 types supported |

### RAG Pipeline| Image Analysis | ✅ | Gemini Vision |

| Export Reports | ✅ | JSON + TXT formats |

1. **User provides**: Crack image + Crack type| Metadata Tracking | ✅ | Topics, sources, types |

2. **Location Analysis**: Gemini analyzes image for crack characteristics

3. **Document Retrieval**: ---

   - Query expansion adds domain synonyms

   - Vector search finds 10 relevant chunks## 🤝 Contributing

   - Re-ranking scores by 6 factors

   - Top 5 chunks selectedTo add more documents:

4. **Cause Analysis**: Gemini determines cause using retrieved context

5. **Report Generation**: Structured output with confidence levels1. Place PDFs in `../RAG_Docs/` folder

2. Run `python process_documents.py`

### Enhanced Retrieval3. System automatically extracts, chunks, and indexes



**Query Expansion Example:**To improve retrieval:

```

"repair diagonal crack" 1. Edit `src/enhanced_retrieval.py`

→ ["repair", "fix", "remediation", "diagonal", "shear", "settlement", "crack", "fissure", ...]2. Adjust query expansions (lines 20-47)

```3. Modify boost factors (lines 99-130)



**Re-ranking Factors:**---

- Exact query match (1.3x boost)

- Multiple term matches (1.1x per term)## 📄 License

- Structural keywords (1.2x boost)

- Crack type match (1.4x boost)This project is for research and educational purposes.

- Document type (standards 1.15x boost)

- Topic relevance (1.25x boost)---



---## 🆘 Need Help?



## 🆘 Quick Reference Commands**Quick Commands:**

```bash

```bash# Test everything

# Setuppython test_enhanced_retrieval.py

pip install -r requirements.txt

echo "GEMINI_API_KEY=your_key" > .env# View statistics

python -c "from src.main_rag import StructuralCrackRAG; print(StructuralCrackRAG().document_manager.get_stats())"

# Usage

python example_usage.py                    # Run example# Run example

python test_enhanced_retrieval.py          # Test systempython example_usage.py

python process_documents.py                # Add new PDFs```



# Verification**Documentation:**

python -c "from src.main_rag import StructuralCrackRAG; print('✓ OK')"- Check docstrings in source files

- See `example_usage.py` for working examples

# Statistics- Review `test_enhanced_retrieval.py` for advanced usage

python -c "from src.main_rag import StructuralCrackRAG; print(StructuralCrackRAG().document_manager.get_stats())"

```---



---**Made with ❤️ for structural engineering analysis**



## 🤝 Support```

crack_analysis_rag/

**Need help?**├── src/

- Check `example_usage.py` for working code examples│   ├── __init__.py

- Review `test_enhanced_retrieval.py` for advanced usage│   ├── crack_types.py          # Crack type definitions

- Read docstrings in source files for API details│   ├── document_manager.py     # RAG document management

│   ├── gemini_analyzer.py      # Gemini AI integration

**Common Questions:**│   └── main_rag.py            # Main system orchestrator

- Q: Do I need to run `process_documents.py`? ├── data/                      # Document storage (auto-created)

  - A: No, 2,626 documents are already processed!├── examples/

- Q: Can I add more documents?│   └── demo.py               # Usage demonstration

  - A: Yes, place PDFs in `../RAG_Docs/` and run the script├── .env                      # API configuration

- Q: How accurate is the system?├── requirements.txt          # Dependencies

  - A: RAG improves accuracy by providing domain-specific context└── README.md                # This file

```

---

## Configuration

**Built for structural engineering analysis with AI-powered precision** 🏗️🤖

### Environment Variables (.env)
```
GOOGLE_API_KEY=your_api_key_here
MODEL_NAME=gemini-2.0-flash-exp
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_TOKENS=4096
TEMPERATURE=0.3
```

## API Reference

### Main Classes

#### `StructuralCrackRAG`
Main system class for crack analysis.

**Methods:**
- `analyze_crack(image_path, crack_type, use_rag=True)`: Full analysis
- `quick_analysis(image_path, crack_type)`: Analysis without RAG
- `add_document(content, crack_type, title)`: Add document to knowledge base
- `get_system_stats()`: Get system statistics
- `export_analysis_results(results, output_file)`: Export analysis results

#### `CrackType`
Enum for crack types: `DIAGONAL`, `STEP`, `VERTICAL`, `HORIZONTAL`, `X_SHAPED`

### Document Management

Add documents to improve analysis accuracy:

```python
# Single document
rag.add_document(
    content="Technical content about cracks...",
    crack_type=CrackType.DIAGONAL,
    title="Document Title",
    source="engineering_manual"
)

# Batch from JSON file
rag.add_documents_from_file("documents.json")
```

### JSON Document Format
```json
[
  {
    "content": "Technical description of crack analysis...",
    "crack_type": "diagonal",
    "title": "Foundation Settlement Study",
    "source": "structural_engineering_handbook"
  }
]
```

## Example Output

The system provides comprehensive analysis including:

- **Location Analysis**: Precise crack positioning and characteristics
- **Cause Determination**: Root cause with engineering explanation
- **Structural Impact**: Assessment of structural implications
- **Recommendations**: Immediate actions and long-term solutions
- **Risk Evaluation**: Safety assessment and monitoring requirements

## Dependencies

- `google-generativeai`: Gemini AI integration
- `sentence-transformers`: Text embeddings
- `faiss-cpu`: Vector similarity search
- `Pillow`: Image processing
- `python-dotenv`: Environment configuration

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `GOOGLE_API_KEY` is set in `.env`
2. **Import Errors**: Install dependencies with `pip install -r requirements.txt`
3. **Image Errors**: Ensure image files exist and are in supported formats (JPG, PNG)
4. **Memory Issues**: For large document collections, consider using `faiss-gpu`

### Performance Tips

- Use smaller images (max 4096px) for faster processing
- Add relevant documents to knowledge base for better analysis
- Use `use_rag=False` for faster analysis when documents aren't available

## Development

To extend the system:

1. **Add New Crack Types**: Extend `CrackType` enum and `CrackCharacteristics`
2. **Custom Models**: Modify `GeminiAnalyzer` to use different models
3. **Enhanced Retrieval**: Improve document retrieval in `CrackRAGDocumentManager`

## License

This project is for educational and research purposes in structural engineering.

## Support

For issues and questions, refer to the code comments and docstrings for detailed implementation guidance.
