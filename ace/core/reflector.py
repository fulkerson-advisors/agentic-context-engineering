"""
Reflector - Generates insights from tool execution traces.

Following ACE paper: Reflector analyzes executions and generates bullet points.
"""

import logging
import uuid
from typing import List, Optional, Literal

from openai import OpenAI
from pydantic import BaseModel

from ace.config import get_openai_model
from ace.core.interfaces import Reflector as ReflectorInterface, ToolExecution, Bullet

logger = logging.getLogger(__name__)

InsightCategory = Literal[
    "success_pattern",
    "error_avoidance",
    "parameter_constraint",
    "edge_case",
]


class InsightItem(BaseModel):
    """Structured insight item returned by the LLM."""

    text: str
    category: InsightCategory


class InsightsList(BaseModel):
    """Pydantic model for structured output."""

    insights: List[InsightItem]


class OpenAIReflector(ReflectorInterface):
    """
    OpenAI-based reflector for generating insights.
    
    Following ACE paper: Uses LLM to analyze tool executions and extract
    reusable patterns, constraints, and error avoidance strategies.
    """
    
    SYSTEM_PROMPT = """You are an expert at analyzing tool execution traces and extracting reusable insights.

Your task is to generate concise, actionable bullet points that capture reusable guidance for future tool calls.

Guidelines:
- Be specific: mention exact parameter names, values, error types
- Be actionable: phrase as instructions (e.g., "Always include X when Y")
- Be concise: one sentence per insight
- Focus on generalizable patterns, not one-off issues
- Prioritize insights that will help future tool calls
Return JSON that matches the schema:
{
  "insights": [
    {
      "text": "Always validate the `unit` argument before calling the API.",
      "category": "success_pattern" | "error_avoidance" | "parameter_constraint" | "edge_case"
    }
  ]
}

Generate 1-3 insights per execution."""
    
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize reflector.
        
        Args:
            client: OpenAI client (creates new if None)
            model: Model to use for reflection. If ``None`` the value is resolved
                from environment variables.
        """
        self.client = client or OpenAI()
        self.model = get_openai_model(
            default="gpt-4.1-mini",
            env_var="OPENAI_REFLECTOR_MODEL",
        ) if model is None else model
    
    async def reflect(
        self,
        execution: ToolExecution,
        conversation_context: Optional[str] = None
    ) -> List[Bullet]:
        """
        Generate bullet points from a tool execution.
        
        Following ACE paper: Analyze execution and extract insights.
        """
        # Build reflection prompt
        prompt = self._build_prompt(execution, conversation_context)
        
        try:
            # Get insights using structured output
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format=InsightsList,
            )
            
            parsed: InsightsList = response.choices[0].message.parsed
            insights = parsed.insights
            
            # Convert to Bullets
            bullets = []
            for item in insights:
                category = item.category
                bullets.append(Bullet(
                    id=str(uuid.uuid4()),
                    content=item.text,
                    tool_name=execution.tool_name,
                    category=category,
                    metadata={
                        'success': execution.success,
                        'timestamp': execution.timestamp.isoformat()
                    }
                ))
            
            return bullets
            
        except Exception as e:
            logger.warning("Reflection failed: %s", e)
            return []
    
    def _build_prompt(
        self,
        execution: ToolExecution,
        context: Optional[str]
    ) -> str:
        """Build reflection prompt from execution."""
        status = "succeeded" if execution.success else "failed"
        
        prompt_parts = [
            "Analyze this tool execution and extract reusable insights.\n",
            f"Tool: {execution.tool_name}",
            f"Arguments: {execution.arguments}",
            f"Status: {status}",
            f"Result: {execution.result[:500]}"
        ]
        
        if execution.error:
            prompt_parts.append(f"Error: {execution.error}")
        
        if context:
            prompt_parts.append(f"\nContext: {context[:500]}")
        
        return "\n".join(prompt_parts)
