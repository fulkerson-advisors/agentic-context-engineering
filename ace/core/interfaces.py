"""
Abstract interfaces for ACE components.

Following the ACE paper architecture with extensible storage backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class Bullet:
    """
    A single insight/bullet point about tool usage.
    
    Following ACE paper: bullets are incremental delta updates to context.
    """
    id: str
    content: str
    tool_name: Optional[str] = None
    category: str = "general"
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ToolExecution:
    """Record of a tool execution for reflection."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    timestamp: datetime = None
    context: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BulletStorage(ABC):
    """
    Abstract interface for bullet storage backends.
    
    Implementations: SQLite, PostgreSQL, MongoDB, etc.
    """
    
    @abstractmethod
    def add_bullet(self, bullet: Bullet) -> None:
        """Add a single bullet to storage."""
        pass
    
    @abstractmethod
    def get_bullets(
        self,
        tool_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Bullet]:
        """Retrieve bullets, optionally filtered by tool."""
        pass
    
    @abstractmethod
    def get_bullet_by_id(self, bullet_id: str) -> Optional[Bullet]:
        """Retrieve a specific bullet by ID."""
        pass
    
    @abstractmethod
    def update_bullet(self, bullet: Bullet) -> None:
        """Update an existing bullet."""
        pass
    
    @abstractmethod
    def delete_bullet(self, bullet_id: str) -> None:
        """Delete a bullet."""
        pass
    
    @abstractmethod
    def get_all_embeddings(self) -> List[tuple[str, List[float]]]:
        """Get all bullet IDs and their embeddings for similarity search."""
        pass


class VectorIndex(ABC):
    """
    Abstract interface for vector similarity search.
    
    Implementations: FAISS, Pinecone, Weaviate, etc.
    """
    
    @abstractmethod
    def add_vectors(self, ids: List[str], vectors: List[List[float]]) -> None:
        """Add vectors to the index."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        threshold: Optional[float] = None
    ) -> List[tuple[str, float]]:
        """
        Search for similar vectors.
        
        Returns list of (id, similarity_score) tuples.
        """
        pass
    
    @abstractmethod
    def remove_vector(self, vector_id: str) -> None:
        """Remove a vector from the index."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Persist index to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk."""
        pass


class Embedder(ABC):
    """
    Abstract interface for text embedding.
    
    Implementations: OpenAI embeddings, Open-source models, etc.
    """
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of embeddings."""
        pass


class Reflector(ABC):
    """
    Abstract interface for generating insights from tool executions.
    
    Following ACE paper: Reflector analyzes execution traces and generates bullets.
    """
    
    @abstractmethod
    async def reflect(
        self,
        execution: ToolExecution,
        conversation_context: Optional[str] = None
    ) -> List[Bullet]:
        """
        Generate bullet points from a tool execution.
        
        Args:
            execution: The tool execution to reflect on
            conversation_context: Optional conversation history
            
        Returns:
            List of generated bullets
        """
        pass


class AgentFramework(ABC):
    """
    Abstract interface for agent frameworks.
    
    Implementations: OpenAI Agents SDK, LangChain, CrewAI, etc.
    """
    
    @abstractmethod
    async def run_with_context(
        self,
        input: str,
        context: str,
        **kwargs
    ) -> Any:
        """
        Run the agent with injected context.
        
        Args:
            input: User input
            context: ACE-generated context to inject
            **kwargs: Framework-specific parameters
            
        Returns:
            Framework-specific result
        """
        pass
    
    @abstractmethod
    def extract_tool_executions(self, result: Any) -> List[ToolExecution]:
        """
        Extract tool execution records from agent result.
        
        Args:
            result: Framework-specific result object
            
        Returns:
            List of tool executions
        """
        pass
