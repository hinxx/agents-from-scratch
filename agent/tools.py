"""
Tool definitions for the agent.

Tools are APIs, not abilities.
The agent requests tools; the system executes them.
"""

from typing import Any


def calculator(a: float, b: float, operation: str = "add") -> float:
    """
    Simple calculator tool.
    
    Args:
        a: First number
        b: Second number
        operation: One of "add", "subtract", "multiply", "divide"
        
    Returns:
        Result of the operation
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float('inf'),
    }
    
    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")
    
    return operations[operation](a, b)


def get_weather(location: str) -> str:
    """
    Get the current weather for a location.
    
    Args:
        location: The city and state/country
        
    Returns:
        A string describing the weather
    """
    # In a real application, you would call a weather API (like OpenWeatherMap) here.
    # For this example, we return a mock response based on a few hardcoded cities.
    mock_weather = {
        "london": "Rainy, 15°C",
        "tokyo": "Sunny, 22°C",
        "new york": "Cloudy, 18°C",
        "paris": "Clear, 20°C"
    }
    
    for city, weather in mock_weather.items():
        if city in location.lower():
            return weather
            
    return f"Sunny, 20°C (Mock weather for {location})"


def get_tool_schema() -> dict:
    """
    Get the schema for available tools.
    
    This is what the agent sees when deciding which tool to call.
    
    Returns:
        Dictionary of tool names to their schemas
    """
    return {
        "calculator": {
            "description": "Perform basic arithmetic operations",
            "parameters": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform"
                }
            },
            "required": ["a", "b"]
        },
        "get_weather": {
            "description": "Get the current weather for a given location",
            "parameters": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    }


def execute_tool(tool_name: str, arguments: dict) -> Any:
    """
    Execute a tool by name with given arguments.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Dictionary of arguments for the tool
        
    Returns:
        Result of the tool execution
        
    Raises:
        ValueError: If tool doesn't exist
    """
    tools = {
        "calculator": calculator,
        "get_weather": get_weather,
    }
    
    if tool_name not in tools:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    return tools[tool_name](**arguments)