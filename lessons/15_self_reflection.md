# Lesson 15 - Self-Reflection and Course Correction

## What Question Are We Answering?

**"How do I get my agent to double-check its work and fix its own mistakes?"**

In earlier lessons, our agent's workflow was purely forward-moving: make a plan, execute a step, move on. The problem with LLMs is that they are probabilistic—sometimes the initial draft or code snippet they generate misses a specific constraint or includes a minor hallucination. 

If the agent just assumes success, those small errors cascade into massive failures later in the pipeline. We solve this by adding a **Self-Reflection** step to the agent's internal loop.

## What You Will Build

An agent routine that:
- Generates an initial response or draft based on a task.
- Pauses and acts as an "Expert Reviewer," critiquing its own draft against the original requirements.
- Evaluates the outcome as either a "pass" or "fail" using structured JSON.
- Automatically feeds failing critiques back into itself to revise the draft (Course Correction).

## New Concepts Introduced

### 1. Observe → Decide → Act → Reflect
We are upgrading the core loop. By forcing the model into a strictly constrained JSON generation where it must produce `"status"` and `"feedback"`, we transition the agent from a passive text generator into an active quality assurance machine.

### 2. Multi-Persona Prompting
You don't need multiple agents (like in Lesson 14) for simple quality checks. You can just execute two different prompts sequentially. First, ask the model to create. Second, give it a different prompt telling it to be a strict evaluator of its own creation. 

## The Code

Look at `agent/agent.py`, specifically `reflect_on_output()` and `run_with_reflection()`:

```python
def reflect_on_output(self, task: str, output: str) -> dict | None:
    """Evaluate an output against its original task."""
    prompt = f"""...
    CRITICAL INSTRUCTIONS:
    1. Respond with ONLY valid JSON
    2. Check for missing requirements, factual errors, or poor formatting
    3. If it perfectly answers the task, status is "pass"
    4. If it needs improvement, status is "fail" and provide specific feedback

    Required JSON format:
    {{"status": "pass" or "fail", "feedback": "specific critique or praise"}}
    ..."""
    # JSON extraction logic
```

The actual course-correction loop uses this critique data to rewrite the output:

```python
def run_with_reflection(self, task: str, max_revisions: int = 3) -> str:
    current_output = self.simple_generate(task)
    
    for attempt in range(1, max_revisions + 1):
        reflection = self.reflect_on_output(task, current_output)
        
        if reflection.get("status") == "pass":
            return current_output
            
        # Re-generate with explicit feedback
        revision_prompt = f"""
        Original Task: {task}
        Current Draft: {current_output}
        Reviewer Feedback: {reflection.get("feedback")}
        
        Provide a completely revised response that fixes all issues..."""
        
        current_output = self.simple_generate(revision_prompt)
        
    return current_output
```

## How to Run

Look at `complete_example.py`, see the `lesson_15_self_reflection()` method:

```python
def lesson_15_self_reflection():
    agent = Agent(MODEL)
    # We use a constrained prompt to force the model to trip up on the first try
    task = "Write a haiku about Python. It MUST explicitly include the word 'Indentation'."
    
    final_output = agent.run_with_reflection(task)
    print(f"\nFinal Approved Output:\n{final_output}")
```

When you run this, you will often see the initial draft fail the constraints (e.g. failing the 5-7-5 syllable rule, or forgetting the word "Indentation"). The agent will print its own failure feedback, course-correct, and spit out an improved final version.

## Key Insights

### Critique is Easier than Creation
LLMs are universally much better at spotting errors in text than they are at writing flawless text from scratch. By separating the creation phase from the critique phase, you leverage this strength perfectly.

### Avoid Infinite Loops
Always set a `max_revisions` limit (like `3` in our code). If a task is literally impossible, or the model gets confused, an agent trying to correct itself forever will waste time and tokens.

### Data-Driven Over Opaque "Thinking"
Notice that the reflection isn't a hidden internal "Chain of Thought". It is an explicit, inspectable JSON dictionary printed right to your terminal. If the agent gets stuck in a loop, you can read exactly *why* the reviewer logic keeps rejecting the draft. 

## Exercises

1. Run the code and observe the feedback. Try adjusting the `task` to be extremely convoluted (e.g., "Write a 3-sentence summary of quantum physics using only words that start with the letter A"). Watch how the agent fails, reflects, and tries to correct itself.
2. Modify `reflect_on_output` so it returns a score out of 10 instead of a pass/fail, and only accept the draft if the score is 8 or higher.

---

**Key Takeaway:** An agent's reliability drastically improves when it acts as its own reviewer. By utilizing structured JSON evaluations, agents can automatically catch and fix their own errors before continuing their pipeline.