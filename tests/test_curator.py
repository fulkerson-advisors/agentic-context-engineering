import importlib.util
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "ace"


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        name,
        PACKAGE_ROOT / relative_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Create lightweight package stubs to avoid importing ace/__init__ (requires numpy/faiss)
ace_module = types.ModuleType("ace")
ace_module.__path__ = [str(PACKAGE_ROOT)]
sys.modules["ace"] = ace_module

ace_core_module = types.ModuleType("ace.core")
ace_core_module.__path__ = [str(PACKAGE_ROOT / "core")]
sys.modules["ace.core"] = ace_core_module

interfaces = _load_module("ace.core.interfaces", "core/interfaces.py")
curator_module = _load_module("ace.core.curator", "core/curator.py")

Curator = curator_module.Curator
Bullet = interfaces.Bullet
BulletStorage = interfaces.BulletStorage
VectorIndex = interfaces.VectorIndex
Embedder = interfaces.Embedder


class FakeStorage(BulletStorage):
    def __init__(self):
        self.bullets = {}
        self.added_bullets = []
        self.updated_bullets = []
        self.deleted_ids = []

    def add_bullet(self, bullet: Bullet) -> None:
        self.bullets[bullet.id] = bullet
        self.added_bullets.append(bullet)

    def get_bullets(self, tool_name=None, limit=None):
        bullets = list(self.bullets.values())
        if tool_name is not None:
            bullets = [b for b in bullets if b.tool_name == tool_name]
        bullets.sort(
            key=lambda b: (b.usage_count, b.created_at),
            reverse=True,
        )
        if limit is not None:
            bullets = bullets[:limit]
        return bullets

    def get_bullet_by_id(self, bullet_id: str):
        return self.bullets.get(bullet_id)

    def update_bullet(self, bullet: Bullet) -> None:
        self.updated_bullets.append(bullet)
        self.bullets[bullet.id] = bullet

    def delete_bullet(self, bullet_id: str) -> None:
        self.deleted_ids.append(bullet_id)
        self.bullets.pop(bullet_id, None)

    def get_all_embeddings(self):
        return [
            (bullet.id, bullet.embedding)
            for bullet in self.bullets.values()
            if bullet.embedding is not None
        ]


class FakeVectorIndex(VectorIndex):
    def __init__(self):
        self.add_vectors_calls = []
        self.search_calls = []
        self.remove_calls = []
        self.save_paths = []
        self.load_paths = []
        self._search_queue = []

    def queue_search_result(self, result):
        self._search_queue.append(result)

    def add_vectors(self, ids, vectors):
        self.add_vectors_calls.append((list(ids), list(vectors)))

    def search(self, query_vector, k=10, threshold=None):
        self.search_calls.append(
            (list(query_vector), k, threshold),
        )
        if self._search_queue:
            return self._search_queue.pop(0)
        return []

    def remove_vector(self, vector_id: str) -> None:
        self.remove_calls.append(vector_id)

    def save(self, path: str) -> None:
        self.save_paths.append(path)

    def load(self, path: str) -> None:
        self.load_paths.append(path)


class FakeEmbedder(Embedder):
    def __init__(self):
        self.calls = []
        self._dimension = 3

    def embed(self, text: str):
        self.calls.append(text)
        length = max(len(text), 1)
        # Simple deterministic embedding vector
        return [
            float(length),
            float(length % 7),
            float(length % 5),
        ]

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]

    def dimension(self) -> int:
        return self._dimension


def test_add_bullets_deduplicates_existing():
    storage = FakeStorage()
    existing = Bullet(
        id="existing",
        content="Always include units for temperature queries.",
        tool_name="weather",
        category="error_avoidance",
        embedding=[0.1, 0.2, 0.3],
        usage_count=1,
    )
    storage.add_bullet(existing)

    vector_index = FakeVectorIndex()
    vector_index.queue_search_result([("existing", 0.95)])
    embedder = FakeEmbedder()

    curator = Curator(
        storage=storage,
        vector_index=vector_index,
        embedder=embedder,
        similarity_threshold=0.9,
    )

    duplicate_candidate = Bullet(
        id="",
        content="Include units when calling the weather tool.",
        tool_name="weather",
        category="error_avoidance",
    )

    added = curator.add_bullets([duplicate_candidate])

    assert added == 0
    updated = storage.get_bullet_by_id("existing")
    assert updated.usage_count == 2
    assert updated.last_used is not None
    assert updated.last_used >= updated.created_at
    # Ensure the embedder was used to populate the missing embedding
    assert embedder.calls == ["Include units when calling the weather tool."]
    # No new vectors should be added beyond the initial load
    assert len(vector_index.add_vectors_calls) == 1


def test_add_bullets_adds_new_when_no_similar():
    storage = FakeStorage()
    vector_index = FakeVectorIndex()
    vector_index.queue_search_result([])
    embedder = FakeEmbedder()

    curator = Curator(
        storage=storage,
        vector_index=vector_index,
        embedder=embedder,
    )

    new_bullet = Bullet(
        id="manual-id",
        content="Retry the request once after transient failures.",
        tool_name="http_client",
        category="success_pattern",
        embedding=[0.3, 0.4, 0.5],
    )

    added = curator.add_bullets([new_bullet])

    assert added == 1
    stored = storage.get_bullet_by_id("manual-id")
    assert stored is not None
    assert stored.content == new_bullet.content
    # The vector index should receive the new embedding (plus potential initial load)
    assert ("manual-id", 0.4) in [
        (ids[0], vectors[0][1]) for ids, vectors in vector_index.add_vectors_calls
    ]


def test_format_bullets_for_prompt_groups_general_and_tool_sections():
    storage = FakeStorage()
    vector_index = FakeVectorIndex()
    embedder = FakeEmbedder()
    curator = Curator(
        storage=storage,
        vector_index=vector_index,
        embedder=embedder,
    )

    general = Bullet(
        id="general-1",
        content="Verify API credentials before making calls.",
        category="parameter_constraint",
        tool_name=None,
        embedding=[0.1, 0.2, 0.3],
    )
    tool_specific = Bullet(
        id="tool-1",
        content="Weather tool expects metric units by default.",
        category="parameter_constraint",
        tool_name="weather",
        embedding=[0.2, 0.3, 0.4],
    )

    formatted = curator.format_bullets_for_prompt([general, tool_specific])

    assert formatted.startswith("## Tool Usage Guidelines")
    assert "### General" in formatted
    assert "- Verify API credentials before making calls." in formatted
    assert "### weather" in formatted
    assert "- Weather tool expects metric units by default." in formatted
