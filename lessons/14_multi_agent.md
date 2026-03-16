# Lesson 14 - Multi-Agent Orchestration - Teamwork, No Magic

## What Question Are We Answering?

**"How do I prevent my agent from getting confused when doing complex, multi-disciplinary tasks?"**

A single agent with a massive system prompt like "You are a world-class researcher, expert programmer, and meticulous QA tester" often suffers from **persona drift**. It forgets its instructions, gets confused by competing priorities, and performs poorly. The solution is specialization: creating multiple agents with narrow, focused prompts and having them coordinate.

## What You Will Build

A simple Multi-Agent architecture consisting of:
- A **Manager Agent** that takes a high-level goal and generates a step-by-step plan (using concepts from Lesson 08).
- A **Worker Agent** with a specialized system prompt that executes those individual steps.
- A mechanism for the manager to delegate execution to the worker.

## New Concepts Introduced

### 1. Agents as Software Components
In many frameworks, multi-agent orchestration is presented as a complex, magical ecosystem where AI entities chat with each other in a virtual room. In reality, an agent is just an instance of a Python class. A "multi-agent system" is simply one object calling a method on another object.

### 2. Persona Segregation
Instead of one mega-prompt, you distribute instructions. The manager only needs to know how to plan. The worker only needs to know how to execute narrow instructions. This isolates context windows and keeps tasks manageable.

## The Code

Look at `agent/agent.py`, specifically the `run_multi_agent()` method:

```python
def run_multi_agent(self, task: str, worker_agent: 'Agent') -> dict | None:
    """
    Manager agent delegates a task to a specialized worker agent.
    """
    # 1. The Manager plans the task
    plan = self.create_plan(task)
    if not plan or "steps" not in plan:
        return None
        
    results = []
    
    # 2. The Manager delegates execution to the Worker
    for step in plan["steps"]:
        worker_response = worker_agent.run(step)
        results.append({"step": step, "worker_result": worker_response})
        
    return {"task": task, "plan": plan, "results": results}
```

Notice the simplicity. The manager isn't "talking" to the worker in an unstructured way. It is simply generating a JSON array (the plan) and passing each string into the worker's standard `run()` method inside a standard `for` loop.

## How to Run

Look at `complete_example.py`, see the `lesson_14_multi_agent()` method:

```python
def lesson_14_multi_agent():
    manager_agent = Agent(MODEL)
    worker_agent = Agent(MODEL)
    
    # Give the worker a highly specialized system prompt
    worker_agent.system_prompt = (
        "You are a diligent worker agent. Execute the specific task "
        "given to you by the manager and return concise, factual results."
    )
    
    results = manager_agent.run_multi_agent("Research AI agents and write a summary", worker_agent)
    print(f"\nOrchestration Results: {results}")
```

Here, you instantiate two distinct `Agent` objects. You override the worker's `system_prompt` to narrow its focus, and then pass it as a parameter to the manager's orchestration function.

## Key Insights

### Multi-Agent is Just Object-Oriented Programming
There is no need for a massive "Agent Swarm" framework to build powerful pipelines. Once you realize an agent is just a function that takes text and returns structured data, orchestrating them becomes a matter of standard programming logic.

### Scalable Complexity
You can easily expand this loop. What if the worker's output is poor? You could instantiate a third `reviewer_agent`, pass the worker's output to it, and if it fails, send it back to the worker in a retry loop.

### Planning vs. Doing
Separating the "planner" from the "doer" is a fundamental architectural pattern in AI. It keeps context windows small and ensures the execution agent stays hyper-focused on the immediate atomic step rather than the overarching goal.

## Exercises

1. **Add a Reviewer:** Modify `run_multi_agent` to take a third agent: `reviewer_agent`. After the worker generates a response, pass both the `step` and `worker_response` to the reviewer. Ask it to output a JSON indicating `"pass"` or `"fail"`.
2. **State Independence:** Use the memory feature (Lesson 07) to see what happens when the manager and the worker have separate memory instances. Notice how isolating their states prevents them from confusing each other's historical context.

---

**Key Takeaway:** Multi-agent orchestration is not magic. It is simply segregating system prompts across multiple object instances and using standard control flow (loops and method calls) to pass data between them.