
"""
Step 1: Basic Invoice Agent
============================
Demonstrates the simplest form of an AI agent that processes a single invoice.
This example shows the foundational pattern for creating and running an agent.

What This Does:
    - Creates an AI agent with basic instructions
    - Processes a single invoice string
    - Returns a summarized response

Enhancement Suggestions:
    1. Add error handling for API failures and network issues
    2. Implement input validation for invoice data
    3. Support batch processing of multiple invoices
    4. Add structured output formatting (JSON, CSV, PDF)
    5. Implement logging for audit trails
    6. Add performance metrics tracking (response time, token usage)
    7. Support different invoice formats (text, JSON, XML)
    8. Add confidence scoring for extracted data
    9. Implement user authentication and authorization
    10. Create a REST API endpoint for web integration
"""

import asyncio
from agent_framework import Agent
from client import get_chat_client

async def main():
    """
    Main function that creates and runs the basic invoice agent.
    
    Enhancement Ideas:
        - Add try-except blocks for error handling
        - Support command-line arguments for invoice input
        - Add timing metrics to track performance
        - Save results to a database or file
    """
    # Create agent with basic summarization instructions
    # The agent uses the Azure OpenAI client configured in client.py
    agent = Agent(
        client=get_chat_client(),
        instructions='Summarize invoice data.'
    )
    
    # Sample invoice data (hardcoded for demo)
    # Enhancement: Read from file, database, or API
    invoice = 'Invoice INV-1001 from Contoso for 1200 USD'
    
    # Run the agent to process the invoice
    # The agent will use the LLM to understand and summarize the invoice
    result = await agent.run(invoice)
    
    # Display the result
    # Enhancement: Format output, save to file, or send to downstream systems
    print(result.text)

# Entry point for the script
if __name__ == '__main__':
    asyncio.run(main())
