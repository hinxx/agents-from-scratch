"""
The Agent - This file grows across all 10 lessons.

This is the heart of the repository. Each lesson adds exactly one capability
to this agent, building understanding progressively.

Lesson progression:
01: Basic LLM chat
02: System prompts (roles)
03: Structured outputs (JSON)
04: Decision-making
05: Tool calling
06: Agent loop
07: Memory
08: Planning
09: Atomic actions
10: AoT (Atom of Thought)
11: Evals (No agent.py changes)
12: Telemetry (No agent.py changes)
13: Human-in-the-Loop (HITL)
14: Multi-Agent Orchestration
15: Self-Reflection
16: Context Management
"""

from typing import Any

from shared.llm import LocalLLM
from shared.utils import extract_json_from_text
from agent.state import AgentState
from agent.memory import Memory
from agent.tools import get_tool_schema, execute_tool
from agent.planner import create_plan, create_atomic_action, create_aot_graph, execute_graph


class Agent:
    """
    An AI agent that grows in capability across lessons.
    
    This is the same agent throughout the repository - it just gains
    new methods and capabilities as lessons progress.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the agent.
        
        Args:
            model_path: Path to the GGUF model file
        """
        # Lesson 01: Basic LLM interaction
        self.llm = LocalLLM(model_path)
        
        # Lesson 02: System prompt for consistent behavior
        self.system_prompt = (
            "You are a calm, precise, and helpful AI assistant. "
            "You explain concepts simply and avoid unnecessary jargon. "
            "You are honest about what you know and don't know."
        )
        
        # Lesson 06: Agent state
        self.state = AgentState()
        
        # Lesson 07: Memory system
        self.memory = Memory()
    
    # ============================================================
    # LESSON 01: Basic LLM Chat
    # ============================================================
    
    def simple_generate(self, user_input: str) -> str:
        """
        Simplest possible interaction - just pass text to the LLM.
        
        Lesson 01 version.
        
        Args:
            user_input: The user's question or request
            
        Returns:
            The model's response
        """
        return self.llm.generate(user_input)
    
    # ============================================================
    # LESSON 02: System Prompts (Roles)
    # ============================================================
    
    def generate_with_role(self, user_input: str) -> str:
        """
        Generate with a system prompt to shape behavior.
        
        Lesson 02 version.
        
        Args:
            user_input: The user's question or request
            
        Returns:
            The model's response with role-based behavior
        """
        # Use a format that doesn't confuse the model
        prompt = f"""{self.system_prompt}

User: {user_input}
Assistant:"""
        
        response = self.llm.generate(prompt)
        # Clean up any potential tag artifacts
        response = response.replace('<SYSTEM>', '').replace('</SYSTEM>', '')
        response = response.replace('<USER>', '').replace('</USER>', '')
        return response.strip()
    
    # ============================================================
    # LESSON 03: Structured Outputs
    # ============================================================
    
    def generate_structured(self, user_input: str, schema: str) -> dict | None:
        """
        Generate structured JSON output with validation and retries.
        
        Lesson 03 version.
        
        Args:
            user_input: The user's question or request
            schema: JSON schema description
            
        Returns:
            Parsed JSON dictionary or None if all retries failed
        """
        prompt = f"""{self.system_prompt}

CRITICAL INSTRUCTIONS:
1. Respond with ONLY valid JSON
2. No explanations, no markdown, no extra text before or after the JSON
3. Start your response with {{ and end with }}

Schema you must follow:
{schema}

User request: {user_input}

Response (JSON only):"""
        
        # Try up to 3 times
        for attempt in range(3):
            response = self.llm.generate(prompt, temperature=0.0)
            parsed = extract_json_from_text(response)
            
            if parsed is not None:
                return parsed
        
        return None
    
    # ============================================================
    # LESSON 04: Decision Making
    # ============================================================
    
    def decide(self, user_input: str, choices: list[str]) -> str | None:
        """
        Make the model choose from a finite set of options.
        
        Lesson 04 version.
        
        Args:
            user_input: The input to make a decision about
            choices: List of possible actions/decisions
            
        Returns:
            The chosen action or None if decision failed
        """
        options = "\n".join(f"- {choice}" for choice in choices)
        
        prompt = f"""{self.system_prompt}

You must choose ONE of the following options. Respond with ONLY valid JSON.

CRITICAL INSTRUCTIONS:
1. Respond with ONLY valid JSON
2. No explanations, no markdown, no other text
3. Start your response with {{ and end with }}

Available choices:
{options}

Required JSON format:
{{"decision": "one_of_the_choices_above"}}

User request: {user_input}

Response (JSON only):"""
        
        for attempt in range(3):
            response = self.llm.generate(prompt, temperature=0.0)
            parsed = extract_json_from_text(response)
            
            if parsed and "decision" in parsed:
                decision = parsed["decision"]
                if decision in choices:
                    return decision
        
        return None
    
    # ============================================================
    # LESSON 05: Tools
    # ============================================================
    
    def request_tool(self, user_input: str) -> dict | None:
        """
        Have the model request a tool call.
        
        Lesson 05 version.
        
        Args:
            user_input: The user's request
            
        Returns:
            Tool call specification or None if request failed
        """
        prompt = f"""{self.system_prompt}

You are a tool-calling assistant. When asked a math question, you must respond with ONLY valid JSON.

Available tool: calculator
- Parameters: a (number), b (number), operation ("add", "subtract", "multiply", or "divide")

CRITICAL INSTRUCTIONS:
1. Respond with ONLY valid JSON
2. No explanations, no markdown, no other text
3. Start your response with {{ and end with }}

Example format:
{{"tool": "calculator", "arguments": {{"a": 42, "b": 7, "operation": "multiply"}}}}

User request: {user_input}

Response (JSON only):"""
        
        for attempt in range(3):
            response = self.llm.generate(prompt, temperature=0.0)
            parsed = extract_json_from_text(response)
            
            if parsed and "tool" in parsed and "arguments" in parsed:
                return parsed
        
        return None
    
    def execute_tool_call(self, tool_call: dict) -> Any:
        """
        Execute a tool call requested by the model.
        
        Args:
            tool_call: Dictionary with "tool" and "arguments"
            
        Returns:
            Result of the tool execution
        """
        return execute_tool(tool_call["tool"], tool_call["arguments"])
    
    # ============================================================
    # LESSON 06: Agent Loop
    # ============================================================
    
    def agent_step(self, user_input: str) -> dict | None:
        """
        Execute one step of the agent loop: observe → decide → act.
        
        Lesson 06 version.
        
        Args:
            user_input: User's input or system observation
            
        Returns:
            Action decision or None if step failed
        """
        state_dict = self.state.to_dict()
        
        prompt = f"""{self.system_prompt}

You are an agent. You must decide the next action and respond with ONLY valid JSON.

Current state: steps={state_dict.get('steps', 0)}, done={state_dict.get('done', False)}

Available actions: analyze, research, summarize, answer, done

CRITICAL INSTRUCTIONS:
1. Respond with ONLY valid JSON
2. No explanations, no markdown, no other text
3. Start your response with {{ and end with }}

Required JSON format:
{{"action": "action_name", "reason": "explanation"}}

User input: {user_input}

Response (JSON only):"""
        
        for attempt in range(3):
            response = self.llm.generate(prompt, temperature=0.0)
            parsed = extract_json_from_text(response)
            
            if parsed and "action" in parsed:
                if "reason" not in parsed:
                    parsed["reason"] = f"Taking action: {parsed['action']}"
                self.state.increment_step()
                return parsed
        
        return None
    
    def run_loop(self, user_input: str, max_steps: int = 5):
        """
        Run the agent loop for multiple steps.
        
        Args:
            user_input: Initial user input
            max_steps: Maximum number of steps to execute
            
        Returns:
            List of action results
        """
        self.state.reset()
        results = []
        
        while not self.state.done and self.state.steps < max_steps:
            action = self.agent_step(user_input)
            
            if action:
                results.append(action)
                
                # Simple termination condition
                if action.get("action") == "done":
                    self.state.mark_done()
            else:
                break
        
        return results
    
    # ============================================================
    # LESSON 07: Memory
    # ============================================================
    
    def run_with_memory(self, user_input: str) -> dict | None:
        """
        Run agent with memory context.
        
        Lesson 07 version.
        
        Args:
            user_input: User's input
            
        Returns:
            Response with potential memory update
        """
        memory_context = self.memory.get_all()
        
        # Build memory context string
        if memory_context:
            memory_str = "You remember the following:\n" + "\n".join(f"- {item}" for item in memory_context)
        else:
            memory_str = "You have no memories yet."
        
        prompt = f"""{self.system_prompt}

You are an agent with memory. You must respond with ONLY valid JSON.

{memory_str}

CRITICAL INSTRUCTIONS:
1. Respond with ONLY valid JSON
2. No explanations, no markdown, no other text
3. Start your response with {{ and end with }}
4. If the user tells you information (like their name), save it to memory
5. If the user asks about something you remember, USE YOUR MEMORY to answer

Required JSON format:
{{"reply": "your response text", "save_to_memory": "fact to remember" or null}}

Examples:
- User says "My name is Alice" → {{"reply": "Nice to meet you, Alice!", "save_to_memory": "User's name is Alice"}}
- User asks "What's my name?" and you remember "User's name is Alice" → {{"reply": "Your name is Alice", "save_to_memory": null}}

User input: {user_input}

Response (JSON only):"""
        
        for attempt in range(3):
            response = self.llm.generate(prompt, temperature=0.0)
            parsed = extract_json_from_text(response)
            
            if parsed and "reply" in parsed:
                # Save to memory if requested
                if parsed.get("save_to_memory"):
                    self.memory.add(parsed["save_to_memory"])
                
                self.state.increment_step()
                return parsed
        
        return None
    
    # ============================================================
    # LESSON 08: Planning
    # ============================================================
    
    def create_plan(self, goal: str) -> dict | None:
        """
        Generate a plan to achieve a goal.
        
        Lesson 08 version.
        
        Args:
            goal: The goal to achieve
            
        Returns:
            Plan with steps
        """
        plan = create_plan(self.llm, goal)
        
        if plan:
            self.state.current_plan = plan
        
        return plan
    
    def execute_plan(self, plan: dict) -> list:
        """
        Execute a plan step by step.
        
        Args:
            plan: Plan dictionary with "steps" list
            
        Returns:
            List of execution results
        """
        if not plan or "steps" not in plan:
            return []
        
        results = []
        
        for step in plan["steps"]:
            # Simple execution - in reality you'd call tools, etc.
            result = {
                "step": step,
                "executed": True
            }
            results.append(result)
            self.state.increment_step()
        
        return results
    
    # ============================================================
    # LESSON 09: Atomic Actions
    # ============================================================
    
    def create_atomic_action(self, step: str) -> dict | None:
        """
        Convert a plan step into an atomic action.
        
        Lesson 09 version.
        
        Atomic actions are the smallest possible actions that can be:
        - Validated independently
        - Tested in isolation
        - Executed safely
        - Rolled back if needed
        
        Args:
            step: A step from a plan (e.g., "Write an explanation of AI agents")
            
        Returns:
            Atomic action dictionary with "action" and "inputs", or None if generation failed
        """
        return create_atomic_action(self.llm, step)
    
    # ============================================================
    # LESSON 10: Atom of Thought (AoT)
    # ============================================================
    
    def create_aot_plan(self, goal: str) -> dict | None:
        """
        Generate an AoT execution graph.
        
        Lesson 10 version.
        
        Args:
            goal: The goal to achieve
            
        Returns:
            AoT graph with atomic nodes and dependencies
        """
        return create_aot_graph(self.llm, goal)
    
    def execute_aot_plan(self, graph: dict) -> list:
        """
        Execute an AoT graph respecting dependencies.
        
        Args:
            graph: AoT graph
            
        Returns:
            List of execution results
        """
        def execute_action(action: str):
            # Placeholder for actual action execution
            return f"Executed: {action}"
        
        return execute_graph(graph, execute_action)
    
    # ============================================================
    # LESSON 13: Human-in-the-Loop (HITL)
    # ============================================================
    
    def run_hitl_loop(self, user_input: str, max_steps: int = 5):
        """
        Run the agent loop but pause for human approval before acting.
        
        Lesson 13 version.
        """
        self.state.reset()
        results = []
        
        while not self.state.done and self.state.steps < max_steps:
            action = self.agent_step(user_input)
            
            if action:
                # The HITL Pause (Intercepting before execution)
                print(f"\n[HITL INTERCEPT] The agent intends to perform: {action.get('action')}")
                print(f"[HITL INTERCEPT] Reason provided: {action.get('reason')}")
                approval = input("[HITL INTERCEPT] Do you approve this action? (y/n): ")
                
                if approval.lower().strip() != 'y':
                    print("[HITL INTERCEPT] Action rejected. Terminating loop.")
                    break
                
                results.append(action)
                
                if action.get("action") == "done":
                    self.state.mark_done()
            else:
                break
        
        return results

    # ============================================================
    # LESSON 14: Multi-Agent Orchestration
    # ============================================================
    
    def run_multi_agent(self, task: str, worker_agent: 'Agent') -> dict | None:
        """
        Manager agent delegates a task to a specialized worker agent.
        
        Lesson 14 version.
        """
        plan = self.create_plan(task)
        if not plan or "steps" not in plan:
            return None
            
        results = []
        
        # The manager delegates the step-by-step execution to the worker
        for step in plan["steps"]:
            worker_response = worker_agent.run(step)
            results.append({"step": step, "worker_result": worker_response})
            
        return {"task": task, "plan": plan, "results": results}

    # ============================================================
    # LESSON 15: Self-Reflection
    # ============================================================
    
    def reflect_on_output(self, task: str, output: str) -> dict | None:
        """
        Evaluate an output against its original task.
        Lesson 15 version.
        """
        prompt = f"""{self.system_prompt}

You are an expert reviewer. Evaluate the following output against the original task.

Task: {task}
Output: {output}

CRITICAL INSTRUCTIONS:
1. Respond with ONLY valid JSON
2. Check for missing requirements, factual errors, or poor formatting
3. If it perfectly answers the task, status is "pass"
4. If it needs improvement, status is "fail" and provide specific feedback

Required JSON format:
{{"status": "pass" or "fail", "feedback": "specific critique or praise"}}

Response (JSON only):"""
        
        for attempt in range(3):
            response = self.llm.generate(prompt, temperature=0.0)
            parsed = extract_json_from_text(response)
            
            if parsed and "status" in parsed and "feedback" in parsed:
                return parsed
                
        return None

    def run_with_reflection(self, task: str, max_revisions: int = 3) -> str:
        """
        Generate, reflect, and revise in a loop.
        Lesson 15 version.
        """
        print(f"Task: {task}\n")
        
        # 1. Initial Draft
        current_output = self.simple_generate(task)
        print(f"[Draft 0]:\n{current_output}\n")
        
        for attempt in range(1, max_revisions + 1):
            # 2. Reflect
            reflection = self.reflect_on_output(task, current_output)
            if not reflection:
                print("[Reflection Failed - Breaking Loop]")
                break
                
            status = reflection.get("status")
            feedback = reflection.get("feedback")
            
            print(f"[Reflection {attempt}]: Status={str(status).upper()}")
            print(f"Feedback: {feedback}\n")
            
            if status == "pass":
                print("Goal achieved!")
                return current_output
                
            # 3. Revise (Course Correction)
            revision_prompt = f"""{self.system_prompt}

You need to revise a draft based on reviewer feedback.

Original Task: {task}
Current Draft: {current_output}
Reviewer Feedback: {feedback}

Provide a completely revised response that fixes all issues mentioned in the feedback."""
            
            current_output = self.simple_generate(revision_prompt)
            print(f"[Draft {attempt}]:\n{current_output}\n")
            
        print("Max revisions reached.")
        return current_output

    # ============================================================
    # LESSON 16: Context Management
    # ============================================================
    
    def summarize_history(self, history: list[str]) -> str | None:
        """
        Summarize a list of past interactions to save context window space.
        Lesson 16 version.
        """
        history_text = "\n".join(history)
        prompt = f"""{self.system_prompt}

You are an expert at information compression. Summarize the following conversation history.
Capture all the key facts, decisions, and outcomes, but make it as concise as possible.

History to summarize:
{history_text}

CRITICAL INSTRUCTIONS:
1. Respond with ONLY valid JSON
2. Required JSON format: {{"summary": "the dense summary text"}}

Response (JSON only):"""
        
        for attempt in range(3):
            response = self.llm.generate(prompt, temperature=0.0)
            parsed = extract_json_from_text(response)
            if parsed and "summary" in parsed:
                return parsed["summary"]
        return None

    def run_with_context_management(self, turns: list[str], max_history: int = 3) -> list[str]:
        """
        Process a long conversation, summarizing old context when it grows too large.
        Lesson 16 version.
        """
        current_history = []
        results = []
        
        for turn in turns:
            print(f"\nUser: {turn}")
            
            # 1. Check if we need to compress context
            if len(current_history) > max_history:
                print(f"[Context Manager] History exceeded {max_history} items. Summarizing older items...")
                # Keep the most recent item, summarize the rest
                to_summarize = current_history[:-1]
                kept_recent = current_history[-1]
                
                summary = self.summarize_history(to_summarize)
                if summary:
                    print(f"[Context Manager] New Compressed Summary: {summary}")
                    current_history = [f"SUMMARY OF PAST: {summary}", kept_recent]
                else:
                    print("[Context Manager] Summarization failed, proceeding with full history.")
            
            # 2. Build the context-aware prompt
            context_str = "\n".join(current_history) if current_history else "No previous context."
            
            prompt = f"""{self.system_prompt}
            
Previous Context:
{context_str}

Current User Request: {turn}

Provide a helpful response based on the context and the request."""
            
            # 3. Generate response
            response = self.simple_generate(prompt)
            print(f"Agent: {response}")
            
            # 4. Add the interaction to history
            current_history.append(f"User: {turn} | Agent: {response}")
            results.append(response)
            
        return results

    # ============================================================
    # MAIN RUN METHOD (evolves across lessons)
    # ============================================================
    
    def run(self, user_input: str) -> str:
        """
        Main entry point for the agent.
        
        This method evolves across lessons to use different capabilities.
        Currently, at: Lesson 07 (with memory)
        
        Args:
            user_input: The user's question or request
            
        Returns:
            The agent's response
        """
        result = self.run_with_memory(user_input)
        
        if result and "reply" in result:
            return result["reply"]
        
        # Fallback to simple generation
        return self.generate_with_role(user_input)