from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SeedData(BaseModel):
    """Data for seeding the context graph with initial nodes."""

    node_id: str
    content: str
    metadata: dict


def _get_datetime_utc() -> datetime:
    """Returns the UTC aware datetime obj"""
    return datetime.now(timezone.utc)


class Node(BaseModel):
    content: str = Field(
        descrition="The content of the NODE",
    )
    importance: float = Field(description="The found importance for a given node")
    created_at: datetime = Field(
        description="The timestamp when the context node was created.",
        default_factory=_get_datetime_utc,
    )


class ContextNode(BaseModel):
    """Data structure for individual nodes in the context graph."""

    node_id: str = Field(
        description="A unique identifier for the context node.",
    )
    content: str = Field(
        description="The content of the context node.",
    )
    metadata: dict[str, float] = Field(
        default_factory=lambda: dict(importance=1.0),
        description="A dictionary of metadata associated with the context node.",
    )
    created_at: datetime = Field(
        description="The timestamp when the context node was created.",
        default_factory=_get_datetime_utc,
    )

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the metadata dictionary by key."""
        return self.metadata.get(key, default)


class ChainOfThought(BaseModel):
    """
    Represents the JSON structure for knowledge graph updates:
        {
            "nodes": { "T1": "desc", "T2": "desc" },
            "edges": [ ["T2","T1"], ... ]
        }
    """

    nodes: dict[str, str]
    edges: list[list[str]]

    @classmethod
    @field_validator("edges")
    def check_edges_not_empty(cls, edges):
        for e in edges:
            if len(e) != 2:
                raise ValueError(f"Edge must have exactly 2 items, got: {e}")
            if not e[0] or not e[1]:
                raise ValueError(f"Edge has empty source/target: {e}")
        return edges
