# Lesson 17 - Dynamic Tool Creation (Code Interpreter)

## What Question Are We Answering?

**"What if I don't know in advance which tools the agent will need?"**

In Lesson 05, we provided the agent with a pre-defined `calculator` tool. However, it is impossible to hardcode every tool an agent might need. What if the user asks the agent to parse a CSV file, scrape a specific website structure, or calculate a complex mathematical formula? 

The solution is **Dynamic Tool Creation**. Instead of giving the agent a predefined function, we give it a Python execution environment and ask it to write its own code on the fly.

## What You Will Build

An agent capability that:
- Takes a complex task (like a math problem).
- Generates a Python script using our strict JSON output methodology.
- Captures the standard output (`stdout`) of the Python process.
- Safely intercepts and executes the code using Python's `exec()` function.

## New Concepts Introduced

### 1. Code as Data
This lesson demystifies features like OpenAI's "Advanced Data Analysis" (formerly Code Interpreter). Under the hood, the LLM is just generating a string of code, a backend server is executing it, and the print statements are returned as observations. 

### 2. Execution Environments (The Sandbox)
Because code is executed dynamically, security is paramount. In this lesson, we use standard Python `exec()`. **In a real production environment**, you would never do this on your main machine. You would execute the generated string inside an isolated Docker container or a WebAssembly (Wasm) sandbox.

## The Code

Look at `agent/agent.py`, specifically `run_dynamic_tool()`:

```python
def run_dynamic_tool(self, task: str) -> dict | None:
    import sys
    import io
    
    prompt = f"""...
    CRITICAL INSTRUCTIONS:
    1. Respond with ONLY valid JSON
    2. Required JSON format: {{"code": "python code string here"}}
    3. The code MUST use print() to output the final answer so it can be captured.
    ..."""
```

Once the agent generates the `code` string, we execute it while intercepting the print statements:

```python
    if parsed and "code" in parsed:
        code = parsed["code"]
        
        # 1. Prepare to intercept print() statements
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        
        error = None
        try:
            # 2. Execute the string as Python code!
            exec(code, {"__builtins__": __builtins__}, {})
        except Exception as e:
            error = str(e)
        finally:
            # 3. Restore standard output
            sys.stdout = original_stdout
        
        output = captured_output.getvalue().strip()
```

## How to Run

Look at `complete_example.py`, see the `lesson_17_dynamic_tools()` method:

```python
def lesson_17_dynamic_tools():
    agent = Agent(MODEL)
    # A math task that LLMs hallucinate, but Python solves perfectly
    agent.run_dynamic_tool("Calculate the 15th Fibonacci number and print it.")
```

When you run this, you will see the LLM generate a `for` loop or recursive function, and the `[Execution Output]` will proudly print `610` (the mathematically correct answer) directly from the script's execution.

## Key Insights

### Escaping LLM Hallucinations
LLMs are notoriously bad at math and deterministic logic. They are text predictors. By giving the LLM the ability to write Python code, you offload the deterministic logic to a deterministic engine (the Python interpreter). The LLM stops predicting the *answer* and starts predicting the *algorithm*.

### Error Recovery (Self-Healing Code)
Because we catch the `Exception` during `exec()`, we could easily pass that error string back to the LLM. Combining this with Lesson 15 (Self-Reflection), you can create an agent that writes a script, encounters a `SyntaxError` or `IndexError`, reads the error traceback, and rewrites its own code until it works.

## Security Warning

Using `exec()` in local applications is dangerous if the LLM decides to write `os.system("rm -rf /")`. While the prompt instructs it to only solve the task, probabilistic models can disobey instructions. 

**Never deploy `exec()` on a production server without rigid, containerized sandboxing.**

## Exercises

1. Run the script and observe the generated code.
2. Modify the task to require a standard library import: `"Generate 5 random numbers between 1 and 100, sort them, and print the result."`
3. *Advanced:* Combine this method with the `run_with_reflection` loop from Lesson 15. If the `exec()` block hits an exception, pass the `error` string back to the LLM and ask it to rewrite the code.

---

**Key Takeaway:** "Code Interpreter" features are just LLMs writing strings and Python executing them. By generating tools dynamically, agents can adapt to entirely new tasks without needing predefined functions.