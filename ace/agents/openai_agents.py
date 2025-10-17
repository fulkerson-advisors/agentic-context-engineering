"""
OpenAI Agents SDK integration for ACE.

Following ACE paper: Inject learned context into agent instructions.
"""

import json
import logging
from typing import Any, Dict, List

from agents import Agent, Runner
from agents.items import ToolCallItem, ToolCallOutputItem
from agents.result import RunResult

from ace.core.interfaces import AgentFramework, ToolExecution

logger = logging.getLogger(__name__)


class OpenAIAgentsFramework(AgentFramework):
    """
    OpenAI Agents SDK integration.
    
    Wraps OpenAI Agents SDK to inject ACE context and extract tool executions.
    """
    
    def __init__(self, agent: Agent):
        """
        Initialize framework wrapper.
        
        Args:
            agent: OpenAI Agent instance
        """
        self.agent = agent
    
    async def run_with_context(
        self,
        input: str,
        context: str,
        **kwargs
    ) -> Any:
        """
        Run agent with injected ACE context.
        
        Args:
            input: User input
            context: ACE-generated context to inject
            **kwargs: Additional parameters for Runner.run
            
        Returns:
            Agent result
        """
        original_instructions = self.agent.instructions
        try:
            # Inject context into instructions
            if isinstance(original_instructions, str):
                if context:
                    self.agent.instructions = (
                        (original_instructions or "") + "\n\n" + context
                        if original_instructions
                        else context
                    )
                else:
                    self.agent.instructions = original_instructions
            elif original_instructions is None and context:
                self.agent.instructions = context
            else:
                # Non-string instructions are left unchanged
                self.agent.instructions = original_instructions

            return await Runner.run(self.agent, input=input, **kwargs)
        finally:
            # Restore original instructions
            self.agent.instructions = original_instructions
    
    def extract_tool_executions(self, result: RunResult) -> List[ToolExecution]:
        """
        Extract tool execution records from agent result.
        
        Args:
            result: OpenAI Agents SDK result
            
        Returns:
            List of tool executions
        """
        if not isinstance(result, RunResult):
            return []

        tool_calls: Dict[str, Dict[str, Any]] = {}
        tool_outputs: Dict[str, List[str]] = {}

        for item in result.new_items:
            if isinstance(item, ToolCallItem):
                raw_call = item.raw_item

                call_id = getattr(raw_call, "call_id", None)
                if not call_id:
                    continue

                if getattr(raw_call, "type", "") != "function_call":
                    continue

                arguments_str = getattr(raw_call, "arguments", "") or ""
                try:
                    arguments = json.loads(arguments_str) if arguments_str else {}
                except json.JSONDecodeError:
                    arguments = {}

                tool_calls[call_id] = {
                    "name": getattr(raw_call, "name", "unknown_tool"),
                    "arguments": arguments,
                }

            elif isinstance(item, ToolCallOutputItem):
                raw_output = item.raw_item
                if isinstance(raw_output, dict):
                    call_id = raw_output.get("call_id")
                else:
                    call_id = getattr(raw_output, "call_id", None)

                if not call_id:
                    continue

                tool_outputs.setdefault(call_id, []).append(item.output or "")

        executions: List[ToolExecution] = []

        for call_id, info in tool_calls.items():
            outputs = tool_outputs.get(call_id, [])
            result_text = "\n".join(o for o in outputs if o)

            success = True
            error = None
            lower_result = result_text.lower()
            if any(keyword in lower_result for keyword in ["error", "failed", "exception"]):
                success = False
                error = result_text or "Tool call reported failure."

            executions.append(
                ToolExecution(
                    tool_name=info["name"],
                    arguments=info["arguments"],
                    result=result_text,
                    success=success,
                    error=error,
                )
            )

        return executions


class ACEAgent:
    """
    ACE-enhanced OpenAI Agent.
    
    Combines OpenAI Agents SDK with ACE for automatic learning.
    """
    
    def __init__(
        self,
        agent: Agent,
        curator,
        reflector,
        enable_learning: bool = True
    ):
        """
        Initialize ACE-enhanced agent.
        
        Args:
            agent: OpenAI Agent instance
            curator: ACE Curator instance
            reflector: ACE Reflector instance
            enable_learning: Whether to enable automatic learning
        """
        self.framework = OpenAIAgentsFramework(agent)
        self.curator = curator
        self.reflector = reflector
        self.enable_learning = enable_learning
    
    async def run(self, input: str, **kwargs) -> Any:
        """
        Run agent with ACE context injection and learning.
        
        Args:
            input: User input
            **kwargs: Additional parameters for agent
            
        Returns:
            Agent result
        """
        # Get relevant bullets
        bullets = self.curator.get_relevant_bullets(
            query=input,
            top_k=10
        )
        
        # Format context
        context = self.curator.format_bullets_for_prompt(bullets)
        
        # Mark bullets as used
        if bullets:
            self.curator.mark_bullets_used([b.id for b in bullets])
        
        # Run agent with context
        result = await self.framework.run_with_context(
            input=input,
            context=context,
            **kwargs
        )
        
        # Learn from executions
        if self.enable_learning:
            await self._learn_from_result(result, input)
        
        return result
    
    async def _learn_from_result(self, result: Any, user_input: str):
        """Learn from agent execution."""
        # Extract tool executions
        executions = self.framework.extract_tool_executions(result)
        
        if not executions:
            return
        
        # Reflect on each execution
        for execution in executions:
            bullets = await self.reflector.reflect(
                execution=execution,
                conversation_context=user_input
            )
            
            if bullets:
                added = self.curator.add_bullets(bullets)
                logger.info("✓ Learned %s new insights from %s", added, execution.tool_name)
                for bullet in bullets:
                    logger.info("  [%s] %s", bullet.category, bullet.content)
    
    def get_stats(self) -> dict:
        """Get learning statistics."""
        return self.curator.get_stats()
