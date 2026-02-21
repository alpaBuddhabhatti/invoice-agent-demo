# Invoice Processing Agent – Azure AI Foundry / Azure OpenAI Demo

A comprehensive demonstration of building AI-powered invoice processing agents using Azure AI Foundry (for model/project management) and Azure OpenAI (for model inference) with the Agent Framework. This project shows progressive complexity from basic agents to multi-tool workflows with business logic.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Step-by-Step Guide](#step-by-step-guide)
- [Enhancement Roadmap](#enhancement-roadmap)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This project demonstrates how to build intelligent invoice processing systems using AI agents. It covers:

- **Basic Agent Creation**: Simple invoice summarization
- **Memory Management**: Conversation context with thread-based memory
- **Tool Integration**: Custom functions for extraction and validation
- **Business Logic**: Approval workflows and conditional processing
- **Azure OpenAI Integration**: Production-ready authentication and configuration

### Use Cases

- Automated invoice data extraction
- Invoice validation and approval workflows
- Conversational invoice queries
- Multi-step invoice processing pipelines
- Compliance and audit trail generation

## 🏗️ Architecture

```
┌─────────────────┐
│   User Input    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Agent Engine   │ ◄─── Instructions & Tools
└────────┬────────┘
         │
         ▼
┌──────────────────────────────┐
│ Azure OpenAI Deployment      │ (managed via Foundry)
│  Chat Client (resource API)  │
└───────────────┬──────────────┘
         │
         ▼
┌─────────────────┐
│  Tool Execution │ (Extract, Validate, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Output      │
└─────────────────┘
```

## 📋 Prerequisites

- **Python**: 3.8 or higher (3.11 recommended)
- **Azure AI Foundry / Azure OpenAI**:
   - A model deployment (often created/managed in Azure AI Foundry)
   - An Azure OpenAI *resource endpoint* + API key to call that deployment
- **Libraries** (installed via `requirements.txt`):
   - `agent-framework`
   - `python-dotenv`
   - `streamlit` (only needed for the Streamlit demos)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd invoice-agent-demo
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**
   - Windows:
     ```powershell
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_API_KEY=your-api-key-here

# Optional: Advanced Configuration
AZURE_OPENAI_API_VERSION=2024-02-01
LOG_LEVEL=INFO
```

Notes:
- `AZURE_OPENAI_DEPLOYMENT` is the preferred variable name. For compatibility, the code also accepts `AZURE_OPENAI_DEPLOYMENT_NAME`.
- The endpoint must be an Azure OpenAI *resource* endpoint (e.g. `https://<resource>.openai.azure.com/`). Azure AI Foundry *project* endpoints are not the same thing and won’t work with the Azure OpenAI chat client used in this repo.

### Security Best Practices

⚠️ **Never commit your API keys to version control!**

- Use environment variables for sensitive data
- Add `.env` to `.gitignore`
- Use Azure Managed Identity in production
- Rotate API keys regularly
- Use Azure Key Vault for secrets management

## 💻 Usage

### Running the Examples

Each step file demonstrates different capabilities:

1. **Basic Agent (Step 1)**
   ```bash
   python step1_basic_agent.py
   ```
   Processes a single invoice and returns a summary.

2. **Thread Memory (Step 2)**
   ```bash
   python step2_thread_memory.py
   ```
   Maintains conversation context for follow-up questions.

3. **Single Tool (Step 3)**
   ```bash
   python step3_invoice_tool.py
   ```
   Uses custom extraction tool for structured data.

4. **Multiple Tools (Step 3 - Alternative)**
   ```bash
   python step3_invoice_tools.py
   ```
   Chains extraction and validation tools together.

5. **Advanced Validation (Step 4)**
   ```bash
   python step4_validation_tool.py
   ```
   Implements business rules and approval thresholds.

## 📁 Project Structure

```
invoice-agent-demo/
├── client.py                   # Azure OpenAI client configuration
├── step1_basic_agent.py        # Basic invoice summarization
├── step2_thread_memory.py      # Conversation memory demo
├── step3_invoice_tool.py       # Single tool usage
├── step3_invoice_tools.py      # Multiple tools workflow
├── step4_validation_tool.py    # Business logic validation
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in repo)
├── .env.example                # Example environment file
└── README.md                   # This file
```

## 📚 Step-by-Step Guide

### Step 1: Basic Agent

**Purpose**: Understand agent creation and basic execution

**Key Concepts**:
- Agent initialization
- Simple instruction-based processing
- Single-turn interaction

**Example**:
```python
agent = Agent(client=get_chat_client(), instructions='Summarize invoice data.')
result = await agent.run('Invoice INV-1001 from Contoso for 1200 USD')
```

### Step 2: Thread Memory

**Purpose**: Maintain conversation context

**Key Concepts**:
- Thread creation and management
- Multi-turn conversations
- Context retention across messages

**Example**:
```python
thread = agent.get_new_thread()
await agent.run('Invoice from Contoso for 1200 USD', thread=thread)
result = await agent.run('What is the total amount?', thread=thread)
```

### Step 3: Tools

**Purpose**: Extend agent capabilities with custom functions

**Key Concepts**:
- Tool decoration and registration
- Function calling by LLM
- Structured data extraction
- Multi-tool orchestration

**Example**:
```python
@tool(name="extract_invoice", description="Extract invoice data")
def extract_invoice(text: str) -> dict:
    return {"vendor": "Contoso", "amount": 1200, "currency": "USD"}

agent = Agent(client=get_chat_client(), tools=[extract_invoice])
```

### Step 4: Business Logic

**Purpose**: Implement real-world validation rules

**Key Concepts**:
- Conditional logic in tools
- Approval workflows
- Threshold-based processing

**Example**:
```python
def validate_invoice(amount: int, currency: str) -> str:
    return "REQUIRES_APPROVAL" if amount > 10000 else "APPROVED"
```

## 🚀 Enhancement Roadmap

### Phase 1: Core Improvements
- [ ] Move API keys to Azure Key Vault
- [ ] Add comprehensive error handling
- [ ] Implement structured logging
- [ ] Add unit and integration tests
- [ ] Create performance benchmarks

### Phase 2: Feature Expansion
- [ ] Real invoice parsing (OCR/PDF)
- [ ] Database integration for persistence
- [ ] Multi-user support with authentication
- [ ] REST API endpoint creation
- [ ] Web UI dashboard

### Phase 3: Advanced Capabilities
- [ ] Multi-currency support with conversion
- [ ] Line-item extraction
- [ ] Duplicate detection
- [ ] Fraud detection with ML
- [ ] Integration with accounting systems (SAP, QuickBooks)

### Phase 4: Enterprise Features
- [ ] Multi-tier approval workflows
- [ ] Email/Slack notifications
- [ ] Comprehensive audit trails
- [ ] Analytics and reporting dashboard
- [ ] Batch processing capabilities
- [ ] High-availability deployment

## ✅ Best Practices

### Security
- Never hardcode API keys
- Use environment variables or Azure Key Vault
- Implement least-privilege access
- Enable audit logging
- Regular security assessments

### Performance
- Implement connection pooling
- Use async/await for concurrent operations
- Cache frequently accessed data
- Monitor token usage and costs
- Implement rate limiting

### Reliability
- Add retry logic with exponential backoff
- Implement circuit breakers
- Comprehensive error handling
- Health checks and monitoring
- Graceful degradation

### Code Quality
- Follow PEP 8 style guidelines
- Write comprehensive docstrings
- Maintain high test coverage
- Use type hints
- Regular code reviews

## 🔧 Troubleshooting

### Common Issues

**Issue**: `401 Unauthorized` error
- **Solution**: Check API key is correct and not expired
- Verify endpoint URL is accurate
- Ensure API key has proper permissions

**Issue**: `404 Resource not found`
- **Solution**: Verify deployment name matches Azure portal
- Check endpoint URL format (no `/openai/v1/` suffix for AzureOpenAIChatClient)

**Issue**: `400 API version not supported`
- **Solution**: Update to supported API version (2024-02-01, 2023-12-01-preview)
- Check Azure portal for available versions

**Issue**: Slow response times
- **Solution**: Monitor token usage
- Optimize prompts for brevity
- Consider using smaller models for simple tasks

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request



---

**Note**: This is a demonstration project. For production use, implement proper security, error handling, and testing.
