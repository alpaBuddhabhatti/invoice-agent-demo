"""
Step 3: Invoice Tool Agent (Single Tool)
=========================================
Demonstrates how to equip an agent with custom tools for specific tasks.
Tools extend the agent's capabilities beyond just text generation.

Key Concepts:
    - Tool: A Python function the agent can call to perform specific actions
    - Tool Decorator: @tool decorator registers functions as agent tools
    - Structured Output: Tools return typed data (dict, str, int, etc.)
    - Function Calling: The LLM decides when to call tools based on instructions

What This Does:
    - Defines a custom invoice extraction tool
    - Registers the tool with the agent
    - Agent automatically calls the tool when needed

Enhancement Suggestions:
    1. Implement real invoice parsing (OCR, PDF extraction)
    2. Add support for multiple invoice formats
    3. Integrate with document storage systems (Azure Blob, S3)
    4. Add data validation and error handling
    5. Extract line items, not just header information
    6. Support multiple currencies and automatic conversion
    7. Add tax calculation functionality
    8. Implement invoice classification (purchase order, receipt, bill)
    9. Add duplicate invoice detection
    10. Integrate with accounting systems (QuickBooks, SAP)
"""

import asyncio
from agent_framework import ChatAgent, tool
from client import get_chat_client

# Define a custom tool for invoice extraction
# The @tool decorator makes this function available to the agent
@tool(
    name="extract_invoice",
    description="Extracts structured invoice data from text or documents"
)
def extract_invoice(text: str) -> dict:
    """
    Extract structured invoice data from text input.
    
    Args:
        text: Raw invoice text or document content
        
    Returns:
        Dictionary containing extracted invoice fields
        
    Enhancement Ideas:
        - Use regex or NLP to parse actual invoice text
        - Integrate OCR for scanned documents (Azure Form Recognizer)
        - Add confidence scores for each extracted field
        - Support multiple invoice formats and templates
        - Extract additional fields (date, invoice number, line items)
        - Validate extracted data against business rules
    """
    # Current implementation: Returns hardcoded data
    # TODO: Replace with actual extraction logic
    return {
        "vendor": "Contoso Ltd",
        "amount": 1200,
        "currency": "USD"
        # Enhancement: Add invoice_number, date, tax_amount, line_items, etc.
    }

async def main():
    """
    Main function demonstrating agent with custom tool.
    
    Enhancement Ideas:
        - Pass actual invoice data/file to the agent
        - Handle tool execution errors gracefully
        - Log tool usage for analytics
        - Support async tools for I/O operations
    """
    # Create agent with extraction instructions and tools
    # The agent knows when to use the extract_invoice tool
    agent = ChatAgent(
        chat_client=get_chat_client(),
        instructions="Extract invoice details accurately.",
        tools=[extract_invoice]  # Register the tool
    )

    # Run the agent with a request that triggers tool usage
    # The agent will automatically call extract_invoice() when appropriate
    result = await agent.run("Extract invoice details")
    
    # Display the result
    # The agent will present the extracted data in a natural language format
    print(result.text)
    
    # Enhancement: Access raw tool results for programmatic use
    # Enhancement: Chain multiple tool calls together

# Entry point for the script
if __name__ == '__main__':
    asyncio.run(main())
