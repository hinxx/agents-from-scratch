# Lesson 13 - Human-in-the-Loop (HITL) - Trust, but Verify

## What Question Are We Answering?

**"How do I prevent an autonomous agent from doing something dangerous?"**

Up to this point, our agent has been fully autonomous. Once you give it a goal, it loops, decides, and executes until it's done. But what if the agent has access to a tool that deletes database records, sends emails to clients, or makes financial transactions? Pure autonomy becomes a liability. 

Human-in-the-Loop (HITL) introduces a pause mechanism, allowing a human to review the agent's intent before the actual execution happens.

## What You Will Build

An intercepted agent loop that:
- Observes the user input and Decides on an action.
- **Pauses** and presents the intended action and reasoning to the user.
- Waits for explicit human approval via standard input.
- Continues execution if approved, or gracefully terminates if rejected.

## New Concepts Introduced

### 1. Execution Interception

In previous lessons, the transition from "Decide" to "Act" was instantaneous. HITL breaks this direct link. The decision (the generated JSON contract) is captured, but the execution function is not called until a blocking operation (like `input()`) is resolved. 

### 2. Autonomy as a Spectrum

Agents do not have to be 100% autonomous. By explicitly tracking state and representing decisions as data structures (JSON), we can slide the scale of autonomy:
- **Fully Autonomous:** Loop runs without interruption.
- **Fully Manual (Chatbot):** Agent suggests one action, stops completely, and waits for a new user prompt.
- **Supervised (HITL):** Agent loops automatically but yields control at critical decision boundaries.

## The Code

Look at `agent/agent.py`, and specifically the `run_hitl_loop()` method:

```python
def run_hitl_loop(self, user_input: str, max_steps: int = 5):
    """
    Run the agent loop but pause for human approval before acting.
    """
    self.state.reset()
    results = []
    
    while not self.state.done and self.state.steps < max_steps:
        # 1. Decide (Generate the intent)
        action = self.agent_step(user_input)
        
        if action:
            # 2. Intercept (The HITL Pause)
            print(f"\n[HITL INTERCEPT] The agent intends to perform: {action.get('action')}")
            print(f"[HITL INTERCEPT] Reason provided: {action.get('reason')}")
            
            # Block execution and wait for human input
            approval = input("[HITL INTERCEPT] Do you approve this action? (y/n): ")
            
            if approval.lower().strip() != 'y':
                print("[HITL INTERCEPT] Action rejected. Terminating loop.")
                break
            
            # 3. Act (Proceed with execution and state update)
            results.append(action)
            
            if action.get("action") == "done":
                self.state.mark_done()
        else:
            break
    
    return results
```

Notice how simple it is. Because we treat the LLM as a software component that outputs a data structure (`action`), inserting a human into the loop only requires standard Python flow control (`print` and `input`). There is no special AI framework magic required to pause an agent.

## How to Run

Look at `complete_example.py`, and see the `lesson_13_hitl()` method:

```python
def lesson_13_hitl():
    print("\nNote: This lesson requires human interaction. The agent will pause for approval.")
    
    agent = Agent(MODEL)
    results = agent.run_hitl_loop("Help me analyze and delete some old files", max_steps=2)
    print(f"\nFinal Execution results: {results}")
```

When you run this, the console will hang, displaying the agent's intent and waiting for you to type 'y' or 'n'.

## Key Insights

### Explicit State Makes Interventions Easy
Because our agent isn't hiding its logic in a massive chain-of-thought string, and instead outputs explicit JSON like `{"action": "delete_files", "reason": "user requested cleanup"}`, we can easily parse and display exactly what it intends to do. 

### You Can Be Selective
In this lesson, we ask for approval on *every* step. In a real-world application, you can easily wrap this logic in a conditional check:

```python
if action.get('action') in ["delete_file", "send_email"]:
    # Require HITL approval
else:
    # Auto-approve safe actions like "read_file" or "calculate"
```

### State Resumption (Advanced)
While our simple `input()` blocks the main thread, this exact same architectural pattern is how asynchronous, server-based agents work. Instead of `input()`, the server saves the agent's `state` and the pending `action` to a database, and the loop terminates. When a human clicks "Approve" on a dashboard web UI hours later, the server loads the state and resumes the loop exactly where it left off.

## Exercises

1. Run the `complete_example.py` script and purposefully reject an action to see the loop terminate safely.
2. Modify `run_hitl_loop` so that instead of just `y/n`, the user can type text to provide "feedback" (e.g., "No, change the target directory to /tmp"), and pass that feedback back into the agent's next observation.

---

**Key Takeaway:** AI agents should not be black boxes that run away with your system. By treating agent loops as standard software loops, you can inject human oversight exactly where you need it using simple programming constructs.