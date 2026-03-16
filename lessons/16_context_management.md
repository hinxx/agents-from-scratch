# Lesson 16 - Context Window Management

## What Question Are We Answering?

**"What happens when my agent runs out of memory/tokens?"**

Every LLM has a "context window"—a strict limit on the number of tokens (words) it can process at one time. In previous lessons, we stored memory and history by blindly appending them to the prompt. If an agent runs a long, complex task, that history will eventually exceed the token limit. When that happens, the model either crashes or suffers from "attention degradation" (forgetting the beginning of the prompt).

To build agents that can run indefinitely, we need a way to manage the size of their context.

## What You Will Build

A context manager that:
- Tracks the size of the agent's conversation history.
- Establishes a "threshold" (e.g., maximum number of interactions).
- Automatically triggers a "Summarization Hook" when the threshold is reached.
- Uses the LLM to compress old history into a dense summary, replacing the old items to save space.

## New Concepts Introduced

### 1. The Context Window
Think of the context window as the agent's short-term working memory. If you overstuff it with verbose logs of past actions, the model gets confused. Managing it is a core software engineering requirement for production AI.

### 2. Compression via Summarization Hook
Instead of just deleting old history (which causes the agent to lose important context), we pause execution to run a specialized prompt. We ask the LLM to read the old history, extract the critical facts, and return a short paragraph. This trades CPU cycles for token space.

## The Code

Look at `agent/agent.py`, specifically `summarize_history()` and `run_with_context_management()`:

```python
def summarize_history(self, history: list[str]) -> str | None:
    """Summarize a list of past interactions to save context window space."""
    history_text = "\n".join(history)
    prompt = f"""...
    You are an expert at information compression. Summarize the following conversation history.
    Capture all the key facts, decisions, and outcomes, but make it as concise as possible.
    ...
    Required JSON format: {{"summary": "the dense summary text"}}"""
    # JSON extraction and retry logic
```

The main function runs a conversation, intercepting the flow to compress state when it gets too large:

```python
def run_with_context_management(self, turns: list[str], max_history: int = 3) -> list[str]:
    current_history = []
    
    for turn in turns:
        # 1. Check Context Size
        if len(current_history) > max_history:
            # Keep the most recent item, summarize the rest
            to_summarize = current_history[:-1]
            kept_recent = current_history[-1]
            
            summary = self.summarize_history(to_summarize)
            if summary:
                # Replace verbose history with dense summary!
                current_history = [f"SUMMARY OF PAST: {summary}", kept_recent]
        
        # 2. Build prompt and generate response normally
        # ...
```

## How to Run

Look at `complete_example.py`, see the `lesson_16_context_management()` method:

```python
def lesson_16_context_management():
    agent = Agent(MODEL)
    
    # Simulate a long conversation
    conversation = [
        "Hi, I'm planning a trip to Japan.",
        "I want to visit Tokyo, Kyoto, and Osaka.",
        "I love eating sushi and visiting ancient temples.",
        "I will be traveling for 14 days in October.",
        "Can you summarize my trip profile so far?"
    ]
    
    # Set max_history to 2 to force a summarization mid-conversation
    agent.run_with_context_management(conversation, max_history=2)
```

When you run this, you will see a print statement explicitly telling you `[Context Manager] History exceeded 2 items`. The agent will output the compressed summary, clearing up its context window while retaining the facts (Japan, Tokyo, Sushi, 14 days) to answer the final question perfectly.

## Key Insights

### Sliding Windows vs Summarization
The naive approach to token limits is a "sliding window" (just deleting the oldest messages). The problem is that early messages often contain critical goals. Summarization hooks give you the best of both worlds: you keep the semantic facts, but discard the verbose syntax.

### Keep the Most Recent Turn Untouched
Notice in the code that we summarize `current_history[:-1]` and explicitly keep `current_history[-1]` untouched. It is a best practice to always leave the immediate prior context exactly as it was written so the flow of the conversation isn't interrupted.

### Token Counting (Advanced)
In this lesson, we use `len(current_history) > max_history` for simplicity. In a real production system, you would use a tokenizer library (like `tiktoken`) to accurately count the exact number of tokens in the prompt, triggering the summary hook only when you hit a strict limit (e.g., `if token_count > 3000`).

## Exercises

1. Run the script and observe the "Compressed Summary" printout. Notice how it condenses multiple lines into one dense fact sheet.
2. Modify the code to use the built-in `Memory` system (from Lesson 07). Have the summarization hook extract facts and save them directly to the key-value memory store instead of keeping a summary string in the prompt.

---

**Key Takeaway:** An agent's memory is finite. By using structured summarization hooks to compress older context, you enable agents to execute indefinitely without hitting token limits or losing focus.