"""
Step 3: Invoice Tools Agent (Multiple Tools)
=============================================
Demonstrates how to create an agent with multiple specialized tools.
This example shows tool chaining where one tool's output can feed into another.

Key Concepts:
    - Multiple Tools: Agents can use many different tools
    - Tool Chaining: Agent orchestrates multiple tool calls in sequence
    - Workflow Automation: Tools work together to complete complex tasks
    - Autonomous Decision Making: Agent decides which tools to use and when

What This Does:
    - Defines extraction and validation tools
    - Agent automatically chains tools together
    - Processes invoice from extraction through validation

Enhancement Suggestions:
    1. Add more tools (calculate_tax, check_duplicate, save_to_db)
    2. Implement approval workflow for high-value invoices
    3. Add email notification tools for stakeholders
    4. Create audit logging tools for compliance
    5. Add payment processing integration
    6. Implement budget checking against department allocations
    7. Add vendor verification tools (check against approved vendor list)
    8. Create reporting and analytics tools
    9. Add document attachment handling tools
    10. Implement multi-currency conversion tools
"""

import asyncio
from agent_framework import Agent, tool
from client import get_chat_client

# Tool 1: Invoice Extraction
@tool(
    name='extract_invoice', 
    description='Extract structured data from invoice text'
)
def extract_invoice(text: str) -> dict:
    """
    Extract key invoice fields from text input.
    
    Args:
        text: Raw invoice text
        
    Returns:
        Dictionary with vendor, amount, and currency
        
    Enhancement Ideas:
        - Parse real invoice data using NLP
        - Extract line items and detailed information
        - Add invoice date, due date, payment terms
        - Include vendor address and contact info
        - Support multiple invoice formats
    """
    # Hardcoded for demo - replace with actual extraction logic
    return {
        'vendor': 'Contoso', 
        'amount': 1200, 
        'currency': 'USD'
    }

# Tool 2: Invoice Validation
@tool(
    name='validate_invoice', 
    description='Validate invoice amount and currency for approval'
)
def validate_invoice(amount: int, currency: str) -> str:
    """
    Validate invoice based on business rules.
    
    Args:
        amount: Invoice amount
        currency: Currency code (USD, EUR, etc.)
        
    Returns:
        Approval status string
        
    Enhancement Ideas:
        - Implement multi-tier approval rules
        - Check against budget constraints
        - Verify currency is supported
        - Cross-check with purchase orders
        - Validate vendor is approved
        - Check for duplicate invoices
        - Apply department-specific rules
        - Add fraud detection checks
    """
    # Simple approval logic - always approves for demo
    # Enhancement: Add real validation rules
    return 'APPROVED'

async def main():
    """
    Main function demonstrating multi-tool agent workflow.
    
    Enhancement Ideas:
        - Add error handling for tool failures
        - Implement conditional tool execution
        - Log tool call sequences for debugging
        - Support parallel tool execution where possible
        - Add tool execution timeout handling
    """
    # Create agent with multiple tools
    # The agent will orchestrate tool usage automatically
    agent = Agent(
        client=get_chat_client(), 
        instructions='Process invoices.', 
        tools=[extract_invoice, validate_invoice]
    )
    
    # Single request triggers multi-step workflow:
    # 1. Agent calls extract_invoice() to get structured data
    # 2. Agent calls validate_invoice() with extracted data
    # 3. Agent synthesizes results into natural language response
    result = await agent.run('Process invoice from Contoso for 1200 USD')
    
    # Display the final result
    print(result.text)
    
    # Enhancement: Access intermediate tool results
    # Enhancement: Implement custom workflow logic

# Entry point for the script
if __name__ == '__main__':
    asyncio.run(main())
