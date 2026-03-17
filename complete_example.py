#!/usr/bin/env python3
"""
Complete Agent Example

This script demonstrates the agent using features from all 12 lessons.
It's meant as a reference for how the pieces fit together.
"""

import time
from agent.agent import Agent

# Using an Ollama model name instead of a file path
MODEL = "llama3.1:8b"

def lesson_01_basic_chat():
    """Lesson 01: Basic LLM interaction"""
    print("\n" + "="*50)
    print("LESSON 01: Basic LLM Chat")
    print("="*50)
    
    agent = Agent(MODEL)
    response = agent.simple_generate("Explain what an AI agent is?")
    print(f"Response: {response}")


def lesson_02_with_role():
    """Lesson 02: System prompts"""
    print("\n" + "="*50)
    print("LESSON 02: With System Prompt")
    print("="*50)
    
    agent = Agent(MODEL)
    response = agent.generate_with_role("Explain what an AI agent is?")
    print(f"Response: {response}")


def lesson_03_structured():
    """Lesson 03: Structured outputs"""
    print("\n" + "="*50)
    print("LESSON 03: Structured Output")
    print("="*50)

    agent = Agent(MODEL)

    schema = """{
  "topic": string,
  "difficulty": "beginner" | "intermediate" | "advanced"
}"""

    result = agent.generate_structured(
        "Explain quantum computing",
        schema
    )
    print(f"Structured result: {result}")


def lesson_04_decisions():
    """Lesson 04: Decision-making"""
    print("\n" + "="*50)
    print("LESSON 04: Decision Making")
    print("="*50)

    agent = Agent(MODEL)

    decision = agent.decide(
        "Can you summarize this article for me?",
        choices=["answer_question", "summarize_text", "translate"]
    )
    print(f"Decision: {decision}")


def lesson_05_tools():
    """Lesson 05: Tool calling"""
    print("\n" + "="*50)
    print("LESSON 05: Tool Calling")
    print("="*50)

    agent = Agent(MODEL)

    tool_call = agent.request_tool("What is 42 * 7?")
    print(f"Tool request: {tool_call}")

    if tool_call:
        result = agent.execute_tool_call(tool_call)
        print(f"Tool result: {result}")

    tool_call = agent.request_tool("What is 69 times 2?")
    print(f"Tool request: {tool_call}")

    if tool_call:
        result = agent.execute_tool_call(tool_call)
        print(f"Tool result: {result}")
        
    print("\n--- Testing the new weather tool ---")
    
    weather_call = agent.request_tool("What's the weather like in Tokyo right now?")
    print(f"Tool request: {weather_call}")

    if weather_call:
        weather_result = agent.execute_tool_call(weather_call)
        print(f"Tool result: {weather_result}")


def lesson_06_agent_loop():
    """Lesson 06: Agent loop"""
    print("\n" + "="*50)
    print("LESSON 06: Agent Loop")
    print("="*50)
    
    agent = Agent(MODEL)
    
    print("\nNote: Repetition in early iterations is expected.")
    print("The agent refines its understanding step by step and may repeat analysis")
    print("before converging on a clearer explanation.\n")
    
    # A query requiring multiple tool uses
    query = "What is the weather in Tokyo, and what is 50 * 3?"
    print(f"User Request: '{query}'\n")
    
    results = agent.run_loop(query, max_steps=4)
    
    for i, result in enumerate(results, 1):
        print(f"Iteration {i}:")
        action = result.get("action", "unknown")
        reason = result.get("reason", "No reason provided")
        arguments = result.get("arguments", {})
        print(f"  Action: {action}")
        if arguments:
            print(f"  Arguments: {arguments}")
        print(f"  Reason: {reason}")
        if i < len(results):
            print()


def lesson_07_memory():
    """Lesson 07: Memory"""
    print("\n" + "="*50)
    print("LESSON 07: Memory")
    print("="*50)

    agent = Agent(MODEL)

    # First interaction - store name
    response1 = agent.run_with_memory("My name is Alice")
    if response1 and "reply" in response1:
        print(f"Response 1: {response1['reply']}")
        if response1.get("save_to_memory"):
            print(f"  → Saved to memory: {response1['save_to_memory']}")
    else:
        print(f"Response 1: {response1}")

    # Second interaction - recall name
    response2 = agent.run_with_memory("What's my name?")
    if response2 and "reply" in response2:
        print(f"Response 2: {response2['reply']}")
        if response2.get("save_to_memory"):
            print(f"  → Saved to memory: {response2['save_to_memory']}")
    else:
        print(f"Response 2: {response2}")

    print(f"\nMemory contents: {agent.memory.get_all()}")


def lesson_08_planning():
    """Lesson 08: Planning"""
    print("\n" + "="*50)
    print("LESSON 08: Planning")
    print("="*50)

    agent = Agent(MODEL)

    plan = agent.create_plan("Write a blog post about AI agents")
    print(f"Plan: {plan}")

    if plan:
        results = agent.execute_plan(plan)
        print(f"Execution results: {results}")


def lesson_09_atomic_actions():
    """Lesson 09: Atomic actions"""
    print("\n" + "="*50)
    print("LESSON 09: Atomic Actions")
    print("="*50)

    agent = Agent(MODEL)

    # Convert a plan step into an atomic action
    step = "Write an explanation of AI agents"
    atomic_action = agent.create_atomic_action(step)
    print(f"Step: {step}")
    print(f"Atomic action: {atomic_action}")

    # Example with a step from a plan
    plan = agent.create_plan("Create a tutorial about Python")
    if plan and "steps" in plan and plan["steps"]:
        first_step = plan["steps"][0]
        atomic_action_from_plan = agent.create_atomic_action(first_step)
        print(f"\nPlan step: {first_step}")
        print(f"Atomic action from plan step: {atomic_action_from_plan}")


def lesson_10_aot():
    """Lesson 10: Atom of Thought"""
    print("\n" + "="*50)
    print("LESSON 10: Atom of Thought")
    print("="*50)

    agent = Agent(MODEL)
    
    graph = agent.create_aot_plan("Research and write article")
    print(f"AoT graph: {graph}")
    
    if graph:
        results = agent.execute_aot_plan(graph)
        print(f"Execution results: {results}")


def lesson_11_evals():
    """Lesson 11: Evals (Regression Testing)"""
    print("\n" + "="*50)
    print("LESSON 11: Evals")
    print("="*50)
    
    from agent.evals import AgentEval, print_eval_report
    from evals.golden_datasets import (
        STRUCTURED_OUTPUT_GOLDEN,
        TOOL_CALL_GOLDEN,
        DECISION_GOLDEN,
        MEMORY_GOLDEN
    )
    
    agent = Agent(MODEL)
    evaluator = AgentEval(agent)
    
    print("\nRunning eval suites...")
    print("(This may take a minute as it runs multiple agent calls)\n")
    
    # Run a subset for demo (full suite can be slow)
    # Using first 2 cases from each suite for quick demo
    results = evaluator.run_all(
        structured_cases=STRUCTURED_OUTPUT_GOLDEN[:2],
        tool_cases=TOOL_CALL_GOLDEN[:2],
        decision_cases=DECISION_GOLDEN[:2],
        memory_cases=MEMORY_GOLDEN[:1]
    )
    
    # Print the report
    print_eval_report(results)
    
    # Show how to access individual results
    print("\nAccessing individual suite results:")
    for suite in results:
        print(f"  {suite.name}: {suite.pass_rate:.0%} pass rate")


def lesson_12_telemetry():
    """Lesson 12: Telemetry (Runtime Observability)"""
    print("\n" + "="*50)
    print("LESSON 12: Telemetry")
    print("="*50)
    
    from agent.telemetry import Telemetry
    
    agent = Agent(MODEL)
    telemetry = Telemetry(log_file="agent_telemetry.jsonl")
    
    # Clear previous telemetry for clean demo
    telemetry.clear()
    
    print("\nRunning agent operations with telemetry...")
    
    # Start a trace for this interaction
    trace_id = telemetry.start_trace()
    print(f"Trace ID: {trace_id}")
    
    # Operation 1: Structured output
    print("\n1. Structured output call...")
    start = time.time()
    result1 = agent.generate_structured(
        "What is Python?", 
        '{"answer": string, "difficulty": "beginner" | "intermediate" | "advanced"}'
    )
    duration1 = (time.time() - start) * 1000
    
    telemetry.log_llm_call(
        prompt_length=150,
        response_length=len(str(result1)) if result1 else 0,
        duration_ms=duration1,
        success=result1 is not None,
        error=None if result1 else "Failed to parse JSON"
    )
    print(f"   Result: {result1}")
    print(f"   Duration: {duration1:.0f}ms")
    
    # Operation 2: Tool call
    print("\n2. Tool call...")
    start = time.time()
    tool_call = agent.request_tool("What is 15 * 8?")
    duration2 = (time.time() - start) * 1000
    
    telemetry.log_llm_call(
        prompt_length=200,
        response_length=len(str(tool_call)) if tool_call else 0,
        duration_ms=duration2,
        success=tool_call is not None
    )
    
    if tool_call:
        telemetry.log_tool_call(
            tool_name=tool_call.get("tool", "unknown"),
            arguments=tool_call.get("arguments", {}),
            result=agent.execute_tool_call(tool_call) if tool_call else None,
            duration_ms=1.0  # Tool execution is fast
        )
        print(f"   Tool: {tool_call}")
    
    # Operation 3: Memory
    print("\n3. Memory operation...")
    start = time.time()
    result3 = agent.run_with_memory("My favorite color is blue")
    duration3 = (time.time() - start) * 1000
    
    telemetry.log_llm_call(
        prompt_length=300,
        response_length=len(str(result3)) if result3 else 0,
        duration_ms=duration3,
        success=result3 is not None
    )
    telemetry.log_memory_operation("add", "favorite color is blue")
    print(f"   Result: {result3}")
    
    # Print telemetry summary
    telemetry.print_summary()
    
    # Show recent spans
    print("\nRecent spans:")
    for span in telemetry.get_recent_spans(5):
        event = span.get("event_type", "unknown")
        duration = span.get("duration_ms", "N/A")
        print(f"  [{event}] duration={duration}ms")
    
    print(f"\nTelemetry logged to: agent_telemetry.jsonl")
    print("View with: cat agent_telemetry.jsonl | head -5")


def lesson_13_hitl():
    """Lesson 13: Human-in-the-Loop (HITL)"""
    print("\n" + "="*50)
    print("LESSON 13: Human-in-the-Loop (HITL)")
    print("="*50)
    
    agent = Agent(MODEL)
    print("\nNote: This lesson requires human interaction. The agent will pause for approval.")
    
    results = agent.run_hitl_loop("Help me analyze and delete some old files", max_steps=2)
    print(f"\nFinal Execution results: {results}")


def lesson_14_multi_agent():
    """Lesson 14: Multi-Agent Orchestration"""
    print("\n" + "="*50)
    print("LESSON 14: Multi-Agent Orchestration")
    print("="*50)
    
    manager_agent = Agent(MODEL)
    worker_agent = Agent(MODEL)
    
    # Give the worker a highly specialized system prompt
    worker_agent.system_prompt = (
        "You are a diligent worker agent. Execute the specific task "
        "given to you by the manager and return concise, factual results."
    )
    
    results = manager_agent.run_multi_agent("Research AI agents and write a summary", worker_agent)
    print(f"\nOrchestration Results: {results}")


def lesson_15_self_reflection():
    """Lesson 15: Self-Reflection"""
    print("\n" + "="*50)
    print("LESSON 15: Self-Reflection and Course Correction")
    print("="*50)
    
    agent = Agent(MODEL)
    
    # Intentionally tricky task to trigger reflection and correction
    task = "Write a haiku about Python. It MUST explicitly include the word 'Indentation'."
    
    final_output = agent.run_with_reflection(task)
    print(f"\nFinal Approved Output:\n{final_output}")


def lesson_16_context_management():
    """Lesson 16: Context Management"""
    print("\n" + "="*50)
    print("LESSON 16: Context Management (Summarization)")
    print("="*50)
    
    agent = Agent(MODEL)
    
    # Simulate a long, multi-turn conversation
    conversation = [
        "Hi, I'm planning a trip to Japan.",
        "I want to visit Tokyo, Kyoto, and Osaka.",
        "I love eating sushi and visiting ancient temples.",
        "I will be traveling for 14 days in October.",
        "Can you summarize my trip profile so far?"
    ]
    
    # Set max_history to 2 to force a summarization mid-conversation
    agent.run_with_context_management(conversation, max_history=2)


def lesson_17_dynamic_tools():
    """Lesson 17: Dynamic Tool Creation"""
    print("\n" + "="*50)
    print("LESSON 17: Dynamic Tool Creation (Code Interpreter)")
    print("="*50)
    
    agent = Agent(MODEL)
    # A math task that LLMs hallucinate, but Python solves perfectly
    agent.run_dynamic_tool("Calculate the 15th Fibonacci number and print it.")


def main():
    """Run all lesson examples"""
    print("\n" + "#"*50)
    print("# AI Agent Examples - All Lessons")
    print("#"*50)
    
    try:
        # Comment out lessons you want to skip
        lesson_01_basic_chat()
        lesson_02_with_role()
        lesson_03_structured()
        lesson_04_decisions()
        lesson_05_tools()
        lesson_06_agent_loop()
        lesson_07_memory()
        lesson_08_planning()
        lesson_09_atomic_actions()
        lesson_10_aot()
        lesson_11_evals()
        lesson_12_telemetry()
        
        # Advanced lessons
        # lesson_13_hitl()  # Commented out by default because it blocks execution via input()
        lesson_14_multi_agent()
        lesson_15_self_reflection()
        lesson_16_context_management()
        lesson_17_dynamic_tools()
        
        print("\n" + "="*50)
        print("All examples completed!")
        print("="*50)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Downloaded a GGUF model")
        print("2. Placed it in the models/ directory")
        print("3. Updated the model path in this script")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()