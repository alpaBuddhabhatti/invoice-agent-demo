# Streamlit Applications Guide

This project includes three different Streamlit applications, each demonstrating different capabilities and approaches to document processing with AI agents.

---

## 📋 Quick Comparison

| App | Best For | Complexity | Multi-Agent | Vision Support | Tool Support |
|-----|----------|------------|-------------|----------------|--------------|
| **Advanced App** | General documents | ⭐ Basic | No | OCR only | No |
| **LLM Extraction** | Complex layouts | ⭐⭐ Medium | No | Full GPT-4V | No |
| **Multi-Agent** | Enterprise workflows | ⭐⭐⭐ Advanced | Yes (4 agents) | Full GPT-4V | Yes |

---

## 1️⃣ streamlit_advanced_app.py

### **Purpose**
General-purpose document analysis with traditional extraction methods.

### **Key Features**
- ✅ Multiple file type support (PDF, images, CSV, Excel, JSON, text)
- ✅ OCR text extraction using EasyOCR
- ✅ Manual parsing for structured files (CSV, Excel, JSON)
- ✅ Single LLM agent for unified Q&A
- ✅ Thread-based conversation memory
- ✅ File preview functionality
- ✅ Export results to JSON/CSV

### **Architecture**
```
User Upload → File Parser → OCR/Manual Extraction → LLM Agent → Results
                                                         ↕
                                                   Thread Memory
```

### **When to Use**
- Basic document analysis needs
- When you have structured data files (CSV, Excel, JSON)
- Budget-conscious projects (uses OCR, less expensive)
- Simple Q&A on document content

### **Limitations**
- OCR may miss complex layouts or handwriting
- No specialized agents for different tasks
- No validation or multi-step workflows
- Requires OCR library installation

### **Run Command**
```bash
streamlit run streamlit_advanced_app.py
```

---

## 2️⃣ streamlit_llm_extraction_app.py

### **Purpose**
LLM-first approach using GPT-4 Vision for intelligent document understanding.

### **Key Features**
- ✅ GPT-4 Vision API for all document types
- ✅ No OCR libraries needed (vision-based)
- ✅ Better handling of complex layouts and tables
- ✅ Semantic understanding of document context
- ✅ Base64 encoding for binary files
- ✅ Multi-file upload support
- ✅ Thread-based conversation across documents
- ✅ Direct text extraction for CSV/JSON files

### **Architecture**
```
User Upload → Base64 Encode → GPT-4 Vision → Intelligent Extraction → Results
                                    ↕
                              Thread Memory
```

### **Supported Files**
- **Vision-based:** PDF, PNG, JPG, JPEG, GIF, BMP, WebP
- **Text-based:** CSV, JSON, TXT

### **When to Use**
- Complex document layouts (invoices, forms, receipts)
- Handwritten or mixed-format documents
- When semantic understanding is critical
- Multi-document analysis with context retention
- When you want the "smartest" extraction

### **Limitations**
- More expensive (uses vision API for everything)
- Slower than traditional parsing
- Token usage can be high with large files
- No specialized validation or workflow steps

### **Run Command**
```bash
streamlit run streamlit_llm_extraction_app.py
```

---

## 3️⃣ streamlit_multi_agent_app.py

### **Purpose**
Enterprise-grade multi-agent system with specialized roles and workflows.

### **Key Features**
- ✅ **4 Specialized Agents:**
  - **ExtractionAgent** - Extracts data (vision + text)
  - **DataAgent** - Structures and normalizes data
  - **ValidationAgent** - Quality assurance and verification
  - **AnalystAgent** - Insights and recommendations
- ✅ Agent hand-off and collaboration
- ✅ Pipeline: Extraction → Validation → Analysis
- ✅ Multi-modal processing (vision, text, data)
- ✅ Confidence scoring and error flagging
- ✅ Detailed pipeline visualization
- ✅ Separate conversation threads per agent

### **Architecture**
```
Upload → ExtractionAgent (vision/text/data) → ValidationAgent → AnalystAgent
             ↓                                      ↓               ↓
        Raw Data                            Quality Score      Insights
```

### **Agent Roles Explained**

#### **ExtractionAgent**
- Handles images, PDFs, text documents
- Uses GPT-4 Vision for visual understanding
- Returns structured JSON with:
  - Document type
  - Confidence score
  - Extracted fields
  - Raw text
  - Notes on ambiguities

#### **DataAgent**
- Takes extracted data and structures it
- Normalizes formats (dates, currencies, numbers)
- Creates tables and relationships
- Fills missing data with flagged defaults
- Converts to requested formats

#### **ValidationAgent**
- Verifies data completeness
- Checks logical consistency
- Validates data types and formats
- Flags suspicious values
- Assigns quality scores (0-100)
- Suggests corrections

#### **AnalystAgent**
- Analyzes patterns and trends
- Provides executive summaries
- Identifies anomalies and outliers
- Makes actionable recommendations
- Compares multiple documents
- Calculates key metrics

### **Workflow Options**

#### **Standard Pipeline** (with validation)
1. Upload file → ExtractionAgent extracts data
2. ValidationAgent checks quality
3. AnalystAgent provides insights
4. User can ask follow-up questions to any agent

#### **Quick Pipeline** (without validation)
1. Upload file → ExtractionAgent extracts data
2. AnalystAgent provides insights
3. User can ask follow-up questions

### **When to Use**
- Enterprise document processing
- When accuracy and validation are critical
- Complex workflows requiring multiple steps
- Batch document processing
- Financial documents requiring verification
- Compliance and audit requirements
- When you need detailed analytics

### **Limitations**
- Most expensive option (4 agents)
- Slower processing (multi-step pipeline)
- More complex to understand and modify
- Overkill for simple extraction tasks

### **Run Command**
```bash
streamlit run streamlit_multi_agent_app.py
```

---

## 🎯 Decision Tree: Which App to Use?

### **Start Here:**

**Do you need validation and compliance?**
- ✅ Yes → **Multi-Agent App**
- ❌ No → Continue...

**Is your document complex (handwritten, mixed layout)?**
- ✅ Yes → **LLM Extraction App**
- ❌ No → Continue...

**Do you have structured data files (CSV/Excel)?**
- ✅ Yes → **Advanced App**
- ❌ No → **LLM Extraction App**

**On a tight budget?**
- ✅ Yes → **Advanced App**
- ❌ No → Consider your complexity needs

---

## 💰 Cost Comparison

### **Advanced App**
- **Cost:** $ (Low)
- OCR runs locally (free after setup)
- LLM only for Q&A, not extraction
- **Best for:** High-volume, budget-conscious

### **LLM Extraction App**
- **Cost:** $$ (Medium)
- Vision API for all binary files
- More tokens per document
- **Best for:** Quality over cost

### **Multi-Agent App**
- **Cost:** $$$ (High)
- 2-4 LLM calls per document
- Validation adds extra tokens
- Agent coordination overhead
- **Best for:** Mission-critical workflows

---

## 🚀 Getting Started

### **Installation**

```bash
# Core requirements (all apps)
pip install streamlit python-dotenv pandas

# For Advanced App (OCR)
pip install easyocr pillow openpyxl

# For LLM/Multi-Agent Apps (Vision)
pip install pillow pdf2image
```

### **Environment Setup**

Create a `.env` file:
```env
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

### **Quick Start**

```bash
# Try the simplest first
streamlit run streamlit_advanced_app.py

# Then test vision capabilities
streamlit run streamlit_llm_extraction_app.py

# Finally explore multi-agent
streamlit run streamlit_multi_agent_app.py
```

---

## 📊 Feature Matrix

| Feature | Advanced | LLM Extraction | Multi-Agent |
|---------|----------|----------------|-------------|
| PDF Support | ✅ | ✅ | ✅ |
| Image Support | ✅ | ✅ | ✅ |
| CSV/Excel Support | ✅ | ✅ | ✅ |
| JSON Support | ✅ | ✅ | ✅ |
| GPT-4 Vision | ❌ | ✅ | ✅ |
| OCR (EasyOCR) | ✅ | ❌ | ❌ |
| Multi-file Upload | ✅ | ✅ | ✅ |
| Conversation Memory | ✅ | ✅ | ✅ (per agent) |
| Data Validation | ❌ | ❌ | ✅ |
| Analytics/Insights | ❌ | ❌ | ✅ |
| Export Results | ✅ | ✅ | ✅ |
| File Preview | ✅ | ✅ | ✅ |
| Confidence Scoring | ❌ | ❌ | ✅ |
| Multi-Agent Pipeline | ❌ | ❌ | ✅ |

---

## 🛠️ Customization Tips

### **For Advanced App:**
- Add custom parsers in `extract_text_from_*()` functions
- Modify OCR settings in `get_ocr_reader()`
- Customize LLM instructions in `get_llm_agent()`

### **For LLM Extraction App:**
- Adjust prompts in `extract_with_llm_vision()`
- Modify thread behavior in `get_thread()`
- Change file encoding in `encode_file_to_base64()`

### **For Multi-Agent App:**
- Customize agent roles in `get_*_agent()` functions
- Modify pipeline in `process_document_multi_agent()`
- Add new agents by creating new agent functions
- Adjust validation thresholds in ValidationAgent instructions

---

## 📝 Best Practices

### **General**
- Always set `.env` variables before running
- Test with small files first
- Monitor token usage in Azure portal
- Use thread memory for multi-turn conversations

### **Advanced App**
- Install OCR dependencies for best results
- Use for high-volume, structured documents
- Cache OCR reader to improve performance

### **LLM Extraction App**
- Use for complex, unstructured documents
- Monitor API costs closely
- Leverage thread memory for document comparisons
- Best with GPT-4 Vision models

### **Multi-Agent App**
- Enable validation for critical documents
- Use analyst agent for business intelligence
- Disable validation for faster processing
- Best for workflows requiring multiple steps

---

## 🐛 Troubleshooting

### **Import Errors**
```bash
# If you see: ImportError: cannot import name 'Agent'
# Solution: Code uses ChatAgent, not Agent (already fixed)
```

### **Vision API Errors**
```bash
# If vision calls fail, check:
# 1. Model supports vision (gpt-4o, gpt-4-turbo, gpt-4o-mini)
# 2. Endpoint is correct in .env
# 3. API key has permissions
```

### **OCR Not Working**
```bash
# Install OCR dependencies:
pip install easyocr pillow
# Note: First run downloads OCR models (~100MB)
```

### **Out of Memory**
```bash
# Reduce image sizes or use LLM Extraction instead of OCR
# OCR is memory-intensive for large images
```

---

## 📚 Related Files

- **`client.py`** - Azure OpenAI client configuration
- **`step*.py`** - Tutorial scripts for learning agent framework
- **`DOCUMENTATION.md`** - Detailed project documentation
- **`ENHANCEMENTS.md`** - Future improvement ideas
- **`README.md`** - Project overview

---

## 🎓 Learning Path

1. **Start with:** `streamlit_advanced_app.py`
   - Understand basic file handling
   - Learn Streamlit UI patterns
   
2. **Move to:** `streamlit_llm_extraction_app.py`
   - See GPT-4 Vision in action
   - Learn LLM-based extraction
   
3. **Master:** `streamlit_multi_agent_app.py`
   - Explore multi-agent patterns
   - Understand enterprise workflows
   
4. **Experiment with:** `step*.py` files
   - Learn agent framework fundamentals
   - Build custom agents and tools

---

*Last updated: February 16, 2026*
