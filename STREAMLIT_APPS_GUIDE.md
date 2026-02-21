# Streamlit Applications Guide

This project includes three different Streamlit applications, each demonstrating different capabilities and approaches to document processing with AI agents.

---

## 📋 Quick Comparison

| App | Best For | Complexity | Multi-Agent | Vision Support | Tool Support |
|-----|----------|------------|-------------|----------------|--------------|
| **Advanced App** | General documents | ⭐ Basic | No | OCR only | No |
| **LLM Extraction** | Prompt-only extraction | ⭐⭐ Medium | No | Base64-in-prompt (best-effort) | No |
| **Multi-Agent** | Enterprise workflows | ⭐⭐⭐ Advanced | Yes (4 agents) | Base64-in-prompt (best-effort) | Yes |

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
LLM-first approach using a single chat agent.

Note: the current implementation does **not** send true image/PDF inputs to a multimodal API. For binary files it includes a **base64 snippet** in the prompt as a best-effort placeholder.

### **Key Features**
- ✅ Single LLM agent for all document types
- ✅ No OCR libraries required (but binary-file accuracy is limited without OCR or true multimodal support)
- ✅ Better handling of complex layouts and tables
- ✅ Semantic understanding of document context
- ✅ Base64 encoding for binary files
- ✅ Multi-file upload support
- ✅ Thread-based conversation across documents
- ✅ Direct text extraction for CSV/JSON files

### **Architecture**
```
User Upload → Base64 Snippet (binary files) → LLM Chat → Extraction → Results
                                    ↕
                              Thread Memory
```

### **Supported Files**
- **Binary files:** PDF, PNG, JPG, JPEG, GIF, BMP, WebP (best-effort via base64 snippet)
- **Text-based:** CSV, JSON, TXT (direct text in prompt)

### **When to Use**
- Complex document layouts (invoices, forms, receipts)
- Handwritten or mixed-format documents
- When semantic understanding is critical
- Multi-document analysis with context retention
- When you want the "smartest" extraction

### **Limitations**
- Not true vision extraction (no multimodal message payloads)
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
- For binary files, uses a base64 snippet in the prompt as a best-effort placeholder (not true vision)
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
- Best-effort binary handling via base64-in-prompt
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
# Recommended (all apps)
pip install -r requirements.txt

# For Advanced App (OCR)
pip install easyocr pillow openpyxl

# For PDF preview
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

# Then test prompt-only LLM extraction
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
| True multimodal vision inputs | ❌ | ❌ | ❌ |
| Base64-in-prompt (binary files) | ❌ | ✅ | ✅ |
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
- Monitor token usage in your provider portal (Foundry/resource metrics)
- Use thread memory for multi-turn conversations

### **Advanced App**
- Install OCR dependencies for best results
- Use for high-volume, structured documents
- Cache OCR reader to improve performance

### **LLM Extraction App**
- Use for complex, unstructured documents
- Monitor API costs closely
- Leverage thread memory for document comparisons
- Best with text-heavy documents unless you add OCR or true multimodal support

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
# Solution: ensure your active virtualenv has `agent-framework` installed and you're running from that environment.
```

### **Binary File Extraction Limitations**
```bash
# If PDFs/images extract poorly:
# 1. Use the Advanced App (OCR) for images
# 2. Convert PDFs to text (or add OCR) before sending to the LLM
# 3. Consider implementing true multimodal message payloads if your client/model supports them
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

- **`client.py`** - Model endpoint client configuration
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
   - See prompt-only LLM extraction
   - Learn how prompting + threads work
   
3. **Master:** `streamlit_multi_agent_app.py`
   - Explore multi-agent patterns
   - Understand enterprise workflows
   
4. **Experiment with:** `step*.py` files
   - Learn agent framework fundamentals
   - Build custom agents and tools

---

*Last updated: February 21, 2026*
