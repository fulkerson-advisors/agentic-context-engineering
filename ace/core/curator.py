"""
Curator - Manages bullet database with semantic deduplication.

Following ACE paper: Curator maintains and retrieves relevant bullets.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from ace.core.interfaces import (
    Bullet,
    BulletStorage,
    VectorIndex,
    Embedder
)


class Curator:
    """
    Curator manages the bullet database with semantic deduplication.
    
    Following ACE paper:
    - Stores bullets with embeddings
    - Deduplicates semantically similar bullets
    - Retrieves relevant bullets for context injection
    """
    
    def __init__(
        self,
        storage: BulletStorage,
        vector_index: VectorIndex,
        embedder: Embedder,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize curator.
        
        Args:
            storage: Bullet storage backend
            vector_index: Vector similarity search index
            embedder: Text embedding model
            similarity_threshold: Threshold for semantic deduplication (0-1)
        """
        self.storage = storage
        self.vector_index = vector_index
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        
        # Load existing embeddings into vector index
        self._init_vector_index()
    
    def _init_vector_index(self):
        """Initialize vector index with existing bullets."""
        embeddings = self.storage.get_all_embeddings()
        if embeddings:
            ids, vectors = zip(*embeddings)
            self.vector_index.add_vectors(list(ids), list(vectors))
    
    def add_bullets(self, bullets: List[Bullet]) -> int:
        """
        Add bullets with semantic deduplication.
        
        Following ACE paper: Use embeddings to detect and merge similar bullets.
        
        Args:
            bullets: List of bullets to add
            
        Returns:
            Number of bullets actually added (after deduplication)
        """
        added_count = 0
        
        for bullet in bullets:
            # Generate embedding
            if bullet.embedding is None:
                bullet.embedding = self.embedder.embed(bullet.content)
            
            # Check for semantic duplicates
            similar = self.vector_index.search(
                bullet.embedding,
                k=5,
                threshold=self.similarity_threshold
            )
            
            if similar:
                # Found semantically similar bullet - increment usage instead
                similar_id, similarity = similar[0]
                existing = self.storage.get_bullet_by_id(similar_id)
                
                if existing:
                    existing.usage_count += 1
                    existing.last_used = datetime.now()
                    self.storage.update_bullet(existing)
                    continue
            
            # No duplicate found - add new bullet
            if bullet.id is None or bullet.id == "":
                bullet.id = str(uuid.uuid4())
            
            self.storage.add_bullet(bullet)
            self.vector_index.add_vectors([bullet.id], [bullet.embedding])
            added_count += 1
        
        return added_count
    
    def get_playbook(
        self,
        tool_name: Optional[str] = None,
    ) -> List[Bullet]:
        """
        Retrieve the current ACE playbook.
        
        Following ACE paper: the Generator is given the full, structured playbook
        (optionally filtered by tool) and decides what guidance to apply.
        
        Args:
            tool_name: Optional tool name filter
            
        Returns:
            Ordered list of bullets (sorted by usage_count DESC, created_at DESC)
        """
        return self.storage.get_bullets(tool_name=tool_name, limit=None)
    
    def get_relevant_bullets(
        self,
        query: Optional[str] = None,
        tool_name: Optional[str] = None,
        top_k: int = 10
    ) -> List[Bullet]:
        """
        Backwards-compatible helper that now proxies to get_playbook.
        
        The ACE paper delegates relevance selection to the Generator itself.
        Consequently, this method simply returns the playbook (optionally limited)
        instead of performing semantic search.
        """
        _ = query  # preserved for API compatibility
        _ = top_k
        return self.get_playbook(tool_name=tool_name)
    
    def format_bullets_for_prompt(self, bullets: List[Bullet]) -> str:
        """
        Format bullets as markdown for injection into prompt.
        
        Following ACE paper: Format as structured guidelines.
        
        Args:
            bullets: List of bullets to format
            
        Returns:
            Formatted markdown string
        """
        if not bullets:
            return ""
        
        lines = ["## Tool Usage Guidelines\n"]
        
        # Group by tool
        by_tool = {}
        general = []
        
        for bullet in bullets:
            if bullet.tool_name:
                by_tool.setdefault(bullet.tool_name, []).append(bullet)
            else:
                general.append(bullet)
        
        # General guidelines first
        if general:
            lines.append("### General")
            for bullet in general:
                lines.append(f"- {bullet.content}")
            lines.append("")
        
        # Tool-specific guidelines
        for tool_name in sorted(by_tool.keys()):
            lines.append(f"### {tool_name}")
            for bullet in by_tool[tool_name]:
                lines.append(f"- {bullet.content}")
            lines.append("")
        
        return "\n".join(lines)
    
    def mark_bullets_used(self, bullet_ids: List[str]):
        """
        Mark bullets as used (increment usage count).
        
        Args:
            bullet_ids: List of bullet IDs that were used
        """
        for bullet_id in bullet_ids:
            bullet = self.storage.get_bullet_by_id(bullet_id)
            if bullet:
                bullet.usage_count += 1
                bullet.last_used = datetime.now()
                self.storage.update_bullet(bullet)
    
    def get_stats(self) -> dict:
        """Get statistics about stored bullets."""
        all_bullets = self.storage.get_bullets()
        
        by_tool = {}
        by_category = {}
        
        for bullet in all_bullets:
            tool = bullet.tool_name or "general"
            by_tool[tool] = by_tool.get(tool, 0) + 1
            by_category[bullet.category] = by_category.get(bullet.category, 0) + 1
        
        # Most used bullets
        sorted_bullets = sorted(all_bullets, key=lambda b: b.usage_count, reverse=True)
        most_used = [
            {
                'id': b.id,
                'content': b.content,
                'tool': b.tool_name,
                'usage_count': b.usage_count
            }
            for b in sorted_bullets[:5]
        ]
        
        return {
            'total_bullets': len(all_bullets),
            'by_tool': by_tool,
            'by_category': by_category,
            'most_used': most_used
        }
