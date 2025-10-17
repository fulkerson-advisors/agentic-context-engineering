"""
SQLite implementation of BulletStorage.
"""

import sqlite3
import json
from typing import List, Optional
from pathlib import Path

from ace.core.interfaces import BulletStorage, Bullet


class SQLiteBulletStorage(BulletStorage):
    """SQLite-backed bullet storage."""
    
    def __init__(self, db_path: str = "ace_bullets.db"):
        """
        Initialize SQLite storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = None
        self._has_embedding_column = False
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bullets (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tool_name TEXT,
                category TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                last_used TEXT,
                usage_count INTEGER DEFAULT 0
            )
        """)
        
        # Create index on tool_name for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_name ON bullets(tool_name)
        """)
        
        self.conn.commit()

        column_info = cursor.execute("PRAGMA table_info(bullets)").fetchall()
        self._has_embedding_column = any(row["name"] == "embedding" for row in column_info)
    
    def add_bullet(self, bullet: Bullet) -> None:
        """Add a bullet to storage."""
        cursor = self.conn.cursor()
        if self._has_embedding_column:
            cursor.execute("""
                INSERT INTO bullets (
                    id, content, tool_name, category, embedding, metadata,
                    created_at, last_used, usage_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bullet.id,
                bullet.content,
                bullet.tool_name,
                bullet.category,
                None,
                json.dumps(bullet.metadata) if bullet.metadata else None,
                bullet.created_at.isoformat(),
                bullet.last_used.isoformat() if bullet.last_used else None,
                bullet.usage_count
            ))
        else:
            cursor.execute("""
                INSERT INTO bullets (
                    id, content, tool_name, category, metadata,
                    created_at, last_used, usage_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bullet.id,
                bullet.content,
                bullet.tool_name,
                bullet.category,
                json.dumps(bullet.metadata) if bullet.metadata else None,
                bullet.created_at.isoformat(),
                bullet.last_used.isoformat() if bullet.last_used else None,
                bullet.usage_count
            ))
        self.conn.commit()
    
    def get_bullets(
        self,
        tool_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Bullet]:
        """Retrieve bullets, optionally filtered."""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM bullets"
        params = []
        
        if tool_name is not None:
            query += " WHERE tool_name = ?"
            params.append(tool_name)
        
        query += " ORDER BY usage_count DESC, created_at DESC"
        
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [self._row_to_bullet(row) for row in rows]
    
    def get_bullet_by_id(self, bullet_id: str) -> Optional[Bullet]:
        """Retrieve a specific bullet."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM bullets WHERE id = ?", (bullet_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return self._row_to_bullet(row)
    
    def update_bullet(self, bullet: Bullet) -> None:
        """Update an existing bullet."""
        cursor = self.conn.cursor()
        if self._has_embedding_column:
            cursor.execute("""
                UPDATE bullets SET
                    content = ?,
                    tool_name = ?,
                    category = ?,
                    embedding = ?,
                    metadata = ?,
                    last_used = ?,
                    usage_count = ?
                WHERE id = ?
            """, (
                bullet.content,
                bullet.tool_name,
                bullet.category,
                None,
                json.dumps(bullet.metadata) if bullet.metadata else None,
                bullet.last_used.isoformat() if bullet.last_used else None,
                bullet.usage_count,
                bullet.id
            ))
        else:
            cursor.execute("""
                UPDATE bullets SET
                    content = ?,
                    tool_name = ?,
                    category = ?,
                    metadata = ?,
                    last_used = ?,
                    usage_count = ?
                WHERE id = ?
            """, (
                bullet.content,
                bullet.tool_name,
                bullet.category,
                json.dumps(bullet.metadata) if bullet.metadata else None,
                bullet.last_used.isoformat() if bullet.last_used else None,
                bullet.usage_count,
                bullet.id
            ))
        self.conn.commit()
    
    def delete_bullet(self, bullet_id: str) -> None:
        """Delete a bullet."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM bullets WHERE id = ?", (bullet_id,))
        self.conn.commit()
    
    def get_all_embeddings(self) -> List[tuple[str, List[float]]]:
        """Get all bullet IDs and embeddings."""
        if not self._has_embedding_column:
            return []

        cursor = self.conn.cursor()
        cursor.execute("SELECT id, embedding FROM bullets WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            embedding = json.loads(row['embedding'])
            result.append((row['id'], embedding))
        
        return result
    
    def _row_to_bullet(self, row: sqlite3.Row) -> Bullet:
        """Convert database row to Bullet object."""
        from datetime import datetime
        
        if self._has_embedding_column:
            embedding_value = row['embedding']
        else:
            embedding_value = None

        return Bullet(
            id=row['id'],
            content=row['content'],
            tool_name=row['tool_name'],
            category=row['category'],
            embedding=json.loads(embedding_value) if embedding_value else None,
            metadata=json.loads(row['metadata']) if row['metadata'] else None,
            created_at=datetime.fromisoformat(row['created_at']),
            last_used=datetime.fromisoformat(row['last_used']) if row['last_used'] else None,
            usage_count=row['usage_count']
        )
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
