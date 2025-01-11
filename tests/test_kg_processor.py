"""Test Knowledge Graph processor functionality."""

import pytest
from donew import DO
import asyncio
import uuid
from typing import Dict, List, Any


@pytest.mark.asyncio
async def test_basic_entity_extraction():
    """Test basic entity extraction through DO.See interface"""
    text = """
    OpenAI CEO Sam Altman has partnered with Microsoft.
    The collaboration was announced in San Francisco.
    """

    result = await DO.Analyze(text)
    assert result is not None

    # Verify entities were extracted
    entities = result.entities()
    assert any(e["text"] == "Sam Altman" and e["label"] == "Person" for e in entities)
    assert any(e["text"] == "OpenAI" and e["label"] == "Company" for e in entities)
    assert any(e["text"] == "Microsoft" and e["label"] == "Company" for e in entities)


@pytest.mark.asyncio
async def test_relationship_extraction():
    """Test relationship extraction between entities"""
    text = """
    Apple Inc., under CEO Tim Cook, has unveiled its latest Vision Pro headset.
    The announcement caused Meta's stock, overseen by Mark Zuckerberg, to drop 5%.
    """

    result = await DO.Analyze(text)

    # Verify relationships were extracted
    relations = result.relations()
    assert any(
        r["source"]["text"] == "Tim Cook"
        and r["target"]["text"] == "Apple Inc."
        and r["type"] == "FOUNDER"
        for r in relations
    )


@pytest.mark.asyncio
async def test_multilingual_support():
    """Test multilingual entity extraction"""
    text = """
    Toyota Motor Corporation, led by CEO Koji Sato in Tokyo, Japan,
    has announced a new partnership with BMW AG in Munich, Germany.
    La collaboration a été annoncée à Paris, France.
    """

    result = await DO.Analyze(text)

    # Verify entities across languages
    entities = result.entities()
    assert any(e["text"] == "Toyota Motor Corporation" for e in entities)
    assert any(e["text"] == "BMW AG" for e in entities)
    assert any(e["text"] == "Tokyo" for e in entities)
    assert any(e["text"] == "Munich" for e in entities)
    assert any(e["text"] == "Paris" for e in entities)


@pytest.mark.asyncio
async def test_complex_document():
    """Test processing of complex documents with multiple entity types"""
    text = """
    The research paper published by Dr. Emily Chen at Stanford University
    describes a breakthrough in quantum computing. The study, funded by
    the National Science Foundation, was conducted in collaboration with
    IBM Research Lab in New York. The results were presented at the
    International Conference on Machine Learning (ICML) 2023 in Hawaii.
    """

    result = await DO.Analyze(text)

    # Verify different entity types
    entities = result.entities()
    entity_labels = {e["label"] for e in entities}
    assert "Person" in entity_labels
    assert "Organization" in entity_labels
    assert "University" in entity_labels
    assert "Conference" in entity_labels
    assert "City" in entity_labels


@pytest.mark.asyncio
async def test_entity_ranking():
    """Test TextRank-based entity importance ranking"""
    text = """
    Alphabet Inc., the parent company of Google, was founded by Larry Page
    and Sergey Brin. The company's current CEO, Sundar Pichai, also leads
    the Google subsidiary. Google, the most prominent subsidiary, operates
    globally with major offices in Mountain View, New York, and London.
    """

    result = await DO.Analyze(text)

    # Verify TextRank results
    rankings = result.textrank()

    # Google should be highly ranked due to multiple mentions
    google_rank = next((r["rank"] for r in rankings if r["text"] == "Google"), None)
    assert google_rank is not None
    assert google_rank > 0.1  # Arbitrary threshold for importance


@pytest.mark.asyncio
async def test_graph_querying():
    """Test graph database querying capabilities"""
    text = """
    DeepMind, acquired by Google in 2014, is headed by Demis Hassabis.
    The AI company collaborates with various research institutions
    including Oxford University. Meanwhile, Google's other AI initiatives
    are led by Jeff Dean, who also oversees Google Brain.
    """

    result = await DO.Analyze(text)

    # Test different query types

    # Find all AI-related companies
    companies = result.query(
        """
        MATCH (e:Entity)
        WHERE e.label = 'Company'
        RETURN e.text as name
    """
    )
    assert len(companies) > 0
    assert any(c["name"] == "DeepMind" for c in companies)

    # Find relationships between people and companies
    relations = result.query(
        """
        MATCH (p:Entity)-[r:Relation]->(o:Entity)
        WHERE p.label = 'Person' AND o.label = 'Company'
        RETURN p.text as person, r.type as relation, o.text as company
    """
    )
    assert len(relations) > 0


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling for various edge cases"""

    # Test empty text
    result = await DO.Analyze("")
    assert result.entities() == []
    assert result.relations() == []

    # Test very short text
    result = await DO.Analyze("Hello world")
    assert len(result.entities()) == 0

    # Test malformed text
    result = await DO.Analyze("!!!@@@###")
    assert len(result.entities()) == 0

    # Test extremely long text
    long_text = "The quick brown fox " * 1000
    result = await DO.Analyze(long_text)
    assert result is not None  # Should handle long text gracefully


@pytest.mark.asyncio
async def test_concurrent_processing():
    """Test concurrent processing of multiple texts"""
    texts = [
        "Apple CEO Tim Cook announced new products.",
        "Microsoft and OpenAI expand partnership.",
        "Google's DeepMind makes AI breakthrough.",
        "Amazon's AWS launches new services.",
    ]

    # Process multiple texts concurrently
    tasks = [DO.Analyze(text) for text in texts]
    results = await asyncio.gather(*tasks)

    # Verify all results
    assert len(results) == len(texts)
    for result in results:
        assert result is not None
        assert len(result.entities()) > 0
