"""Knowledge Graph module for entity and relationship extraction.

This implementation is inspired by and adapted from:
GraphGeeks.org talk 2024-08-14 https://live.zoho.com/PBOB6fvr6c
Original source: https://raw.githubusercontent.com/DerwenAI/strwythura/refs/heads/main/demo.py

The Knowledge Graph implementation uses:
- GLiNER for Named Entity Recognition
- GLiREL for Relationship Extraction
- KuzuDB for graph storage and querying
- spaCy for text processing and chunking

The graph is constructed in layers:
1. Base layer: Textual analysis using spaCy parse trees
2. Entity layer: Named entities and noun chunks from GLiNER
3. Relationship layer: Semantic relationships from GLiREL
4. Storage layer: Persistent graph storage in KuzuDB
"""

from collections import defaultdict
from dataclasses import dataclass
import itertools
import logging
import math
import os
import pathlib
import unicodedata
from typing import Dict, List, Optional, Union, Set, Tuple

import glirel
import spacy
from spacy.language import Language
from gliner_spacy.pipeline import GlinerSpacy
import transformers
import networkx as nx
import kuzu

NER_LABELS: List[str] = [
    "Behavior",
    "City",
    "Company",
    "Condition",
    "Conference",
    "Country",
    "Food",
    "Food Additive",
    "Hospital",
    "Organ",
    "Organization",
    "People Group",
    "Person",
    "Publication",
    "Research",
    "Science",
    "University",
]

# Global constants for GLiREL
RE_LABELS: dict = {
    "glirel_labels": {
        "co_founder": {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},
        "country_of_origin": {
            "allowed_head": ["PERSON", "ORG"],
            "allowed_tail": ["LOC", "GPE"],
        },
        "no_relation": {},
        "parent": {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},
        "followed_by": {
            "allowed_head": ["PERSON", "ORG"],
            "allowed_tail": ["PERSON", "ORG"],
        },
        "spouse": {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},
        "child": {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},
        "founder": {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},
        "headquartered_in": {
            "allowed_head": ["ORG"],
            "allowed_tail": ["LOC", "GPE", "FAC"],
        },
        "acquired_by": {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "PERSON"]},
        "subsidiary_of": {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "PERSON"]},
    }
}

STOP_WORDS: Set[str] = set(
    [
        "PRON.it",
        "PRON.that",
        "PRON.they",
        "PRON.those",
        "PRON.we",
        "PRON.which",
        "PRON.who",
    ]
)


@dataclass(order=False, frozen=False)
class Entity:
    """Entity class for storing extracted entities."""

    loc: Tuple[int]  # Location in text
    key: str  # Unique identifier
    text: str  # Original text
    label: str  # Entity type
    chunk_id: int  # Which text chunk
    sent_id: int  # Which sentence
    span: spacy.tokens.span.Span  # Original spaCy span
    node: Optional[int] = None  # Graph node ID


class TextChunk:
    """Text chunk for processing."""

    def __init__(self, uid: int, text: str):
        self.uid = uid
        self.text = text


class KnowledgeGraph:
    """Knowledge graph with KuzuDB storage."""

    # Constants and configuration
    CHUNK_SIZE: int = 1024
    GLINER_MODEL: str = "urchade/gliner_small-v2.1"
    SPACY_MODEL: str = "en_core_web_md"

    # Entity labels we want to extract

    def __init__(self, db_path: Optional[str] = None):
        """Initialize KG with KuzuDB storage.

        Args:
            db_path: Path to KuzuDB database. If None, uses in-memory mode.
                    If ":memory:", uses in-memory mode explicitly.
                    Otherwise uses on-disk mode at the specified path.
        """
        # Initialize NLP pipeline
        self._nlp = self._init_nlp()

        # Initialize KuzuDB
        # Initialize database in memory or on disk
        db_path = ":memory:" if db_path in (None, ":memory:") else db_path
        self._db = kuzu.Database(db_path)

        self._conn = kuzu.Connection(self._db)
        self._init_db()

    def _init_nlp(self) -> Language:
        """Initialize NLP pipeline with spaCy, GLiNER, and GLiREL."""
        # Disable noisy logging
        logging.disable(logging.ERROR)
        transformers.logging.set_verbosity_error()
        os.environ["TOKENIZERS_PARALLELISM"] = "0"

        # Load spaCy model
        nlp = spacy.load(self.SPACY_MODEL)

        # Add GLiNER
        nlp.add_pipe(
            "gliner_spacy",
            config={
                "gliner_model": self.GLINER_MODEL,
                "labels": NER_LABELS,
                "chunk_size": self.CHUNK_SIZE,
                "style": "ent",
            },
        )

        # Add GLiREL exactly like demo.py
        nlp.add_pipe("glirel", after="ner")

        return nlp

    def _init_db(self) -> None:
        """Initialize KuzuDB schema."""
        if not self._conn:
            return

        try:
            # Create Entity node table with proper schema
            self._conn.execute(
                """
                CREATE NODE TABLE IF NOT EXISTS Entity(
                    text STRING,
                    label STRING,
                    PRIMARY KEY (text, label)
                )
            """
            )

            # Create relationship table with proper schema
            self._conn.execute(
                """
                CREATE REL TABLE IF NOT EXISTS Relation(
                    FROM Entity TO Entity,
                    type STRING,
                    prob FLOAT
                )
            """
            )
        except Exception as e:
            # Log error but continue since tables may already exist
            logging.error(f"Error initializing database: {e}")
            pass

    def _uni_scrubber(self, text: str) -> str:
        """Clean and normalize Unicode text."""
        if not isinstance(text, str):
            print("not a string?", type(text), text)
            return ""

        # Join lines and normalize whitespace
        limpio = " ".join(map(lambda s: s.strip(), text.split("\n"))).strip()

        # Replace smart quotes and apostrophes
        limpio = limpio.replace('"', '"').replace('"', '"')
        limpio = (
            limpio.replace("'", "'")
            .replace("'", "'")
            .replace("`", "'")
            .replace("â", "'")
        )

        # Replace other special characters
        limpio = limpio.replace("…", "...").replace("–", "-")

        # Normalize to ASCII
        limpio = str(
            unicodedata.normalize("NFKD", limpio)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )

        return limpio

    def _make_chunk(
        self, doc: spacy.tokens.doc.Doc, chunk_list: List[TextChunk], chunk_id: int
    ) -> int:
        """Split document into text chunks."""
        chunks: List[str] = []
        chunk_total: int = 0
        prev_line: str = ""

        for sent_id, sent in enumerate(doc.sents):
            line: str = self._uni_scrubber(sent.text)
            line_len: int = len(line)

            if (chunk_total + line_len) > self.CHUNK_SIZE:
                # Emit current chunk
                chunk_list.append(TextChunk(uid=chunk_id, text="\n".join(chunks)))
                # Start new chunk
                chunks = [prev_line, line]
                chunk_total = len(prev_line) + line_len
                chunk_id += 1
            else:
                # Append to current chunk
                chunks.append(line)
                chunk_total += line_len
            prev_line = line

        # Emit trailing chunk
        chunk_list.append(TextChunk(uid=chunk_id, text="\n".join(chunks)))
        return chunk_id + 1

    def _make_entity(
        self,
        span_decoder: Dict[tuple, Entity],
        sent_map: Dict[spacy.tokens.span.Span, int],
        span: spacy.tokens.span.Span,
        chunk: TextChunk,
        *,
        debug: bool = False,
    ) -> None:
        """Create an Entity from a spaCy span."""
        sent = span.sent
        sent_id = sent_map.get(sent, -1)
        key = f"{span.label_}.{span.text}"

        ent = Entity(
            loc=(span.start, span.end),
            key=key,
            text=span.text,
            label=span.label_,
            chunk_id=chunk.uid,
            sent_id=sent_id,
            span=span,
        )

        span_decoder[ent.loc] = ent

    def _extract_entity(
        self,
        known_lemma: List[str],
        lex_graph: nx.Graph,
        ent: Entity,
        *,
        debug: bool = False,
    ) -> None:
        """Extract entity and add to lexical graph."""
        if ent.node is not None:
            return

        node_id = len(lex_graph)
        ent.node = node_id

        lex_graph.add_node(
            node_id,
            key=ent.key,
            text=ent.text,
            label=ent.label,
            kind="Entity",
            chunk=ent.chunk_id,
            count=1,
        )

        if ent.key not in known_lemma:
            known_lemma.append(ent.key)

    def _extract_relations(
        self,
        known_lemma: List[str],
        lex_graph: nx.Graph,
        span_decoder: Dict[tuple, Entity],
        sent_map: Dict[spacy.tokens.span.Span, int],
        doc: spacy.tokens.doc.Doc,
        chunk: TextChunk,
        *,
        debug: bool = False,
    ) -> None:
        """Extract relations inferred by GLiREL adding these to the graph."""
        relations: List[dict] = sorted(
            doc._.relations,
            key=lambda item: item["score"],
            reverse=True,
        )

        for item in relations:
            src_loc: Tuple[int] = tuple(item["head_pos"])
            dst_loc: Tuple[int] = tuple(item["tail_pos"])
            redact_rel: bool = False

            if src_loc not in span_decoder:
                if debug:
                    print("MISSING src entity:", item["head_text"], item["head_pos"])
                continue

            if dst_loc not in span_decoder:
                if debug:
                    print("MISSING dst entity:", item["tail_text"], item["tail_pos"])
                continue

            src_ent = span_decoder[src_loc]
            dst_ent = span_decoder[dst_loc]

            rel: str = item["label"].strip().replace(" ", "_").upper()
            prob: float = round(item["score"], 3)

            if debug:
                print(f"{src_ent.text} -> {rel} -> {dst_ent.text} | {prob}")

            lex_graph.add_edge(
                src_ent.node,
                dst_ent.node,
                rel=rel,
                prob=prob,
            )

    def _connect_entities(
        self,
        lex_graph: nx.Graph,
        span_decoder: Dict[tuple, Entity],
    ) -> None:
        """Connect entities which co-occur within the same sentence."""
        ent_map: Dict[int, Set[int]] = defaultdict(set)

        for ent in span_decoder.values():
            if ent.node is not None:
                ent_map[ent.sent_id].add(ent.node)

        for sent_id, nodes in ent_map.items():
            for pair in itertools.combinations(list(nodes), 2):
                if not lex_graph.has_edge(*pair):
                    lex_graph.add_edge(
                        pair[0],
                        pair[1],
                        rel="CO_OCCURS_WITH",
                        prob=1.0,
                    )

    def analyze(self, text: str, *, debug: bool = False) -> Dict:
        """Analyze text to extract entities and relationships."""
        # Initialize processing structures
        chunk_list: List[TextChunk] = []
        lex_graph: nx.Graph = nx.Graph()
        known_lemma: List[str] = []

        # Split text into chunks and process with GLiREL labels
        doc: spacy.tokens.doc.Doc = list(
            self._nlp.pipe(
                [(text, RE_LABELS)],
                as_tuples=True,
            )
        )[0][0]

        self._make_chunk(doc, chunk_list, 0)

        # Process each chunk
        for chunk in chunk_list:
            # Parse the chunk with GLiREL labels
            doc = list(
                self._nlp.pipe(
                    [(chunk.text, RE_LABELS)],
                    as_tuples=True,
                )
            )[
                0
            ][0]

            # Scan document tokens to add lemmas to lexical graph using textrank
            for sent in doc.sents:
                node_seq: List[int] = []

                if debug:
                    print(f"Processing sentence: {sent}")

                for tok in sent:
                    text: str = tok.text.strip()

                    if tok.pos_ in ["NOUN", "PROPN"]:
                        key: str = tok.pos_ + "." + tok.lemma_.strip().lower()
                        prev_known: bool = False

                        if key not in known_lemma:
                            # Create new node
                            known_lemma.append(key)
                        else:
                            # Link to existing node, adding weight
                            prev_known = True

                        node_id: int = known_lemma.index(key)
                        node_seq.append(node_id)

                        if not lex_graph.has_node(node_id):
                            lex_graph.add_node(
                                node_id,
                                key=key,
                                kind="Lemma",
                                pos=tok.pos_,
                                text=text,
                                chunk=chunk.uid,
                                count=1,
                            )
                        elif prev_known:
                            node: dict = lex_graph.nodes[node_id]
                            node["count"] += 1

                # Create textrank edges for lexical graph
                if debug:
                    print(f"Node sequence: {node_seq}")

                for hop in range(3):  # TR_LOOKBACK = 3
                    for node_id, node in enumerate(node_seq[: -1 - hop]):
                        neighbor: int = node_seq[hop + node_id + 1]

                        if not lex_graph.has_edge(node, neighbor):
                            lex_graph.add_edge(
                                node,
                                neighbor,
                                rel="FOLLOWS_LEXICALLY",
                            )

            # Track sentence numbers
            sent_map: Dict[spacy.tokens.span.Span, int] = {}
            for sent_id, sent in enumerate(doc.sents):
                sent_map[sent] = sent_id

            # Process spans
            span_decoder: Dict[tuple, Entity] = {}
            for span in doc.ents:  # Only process named entities, not noun chunks
                self._make_entity(span_decoder, sent_map, span, chunk, debug=debug)

            # Extract entities and relations
            for ent in span_decoder.values():
                if ent.key not in STOP_WORDS:  # Add stop word check like demo.py
                    self._extract_entity(known_lemma, lex_graph, ent, debug=debug)
            self._extract_relations(
                known_lemma, lex_graph, span_decoder, sent_map, doc, chunk, debug=debug
            )
            # Connect co-occurring entities
            self._connect_entities(lex_graph, span_decoder)

        # Convert to return format
        entities = []
        relations = []

        for node_id, node_data in lex_graph.nodes(data=True):
            if node_data.get("kind") == "Entity":
                entities.append(
                    {
                        "text": node_data["text"],
                        "label": node_data["label"],
                    }
                )

        for src, dst, edge_data in lex_graph.edges(data=True):
            src_data = lex_graph.nodes[src]
            dst_data = lex_graph.nodes[dst]
            if src_data.get("kind") == "Entity" and dst_data.get("kind") == "Entity":
                relations.append(
                    {
                        "source": {
                            "text": src_data["text"],
                            "label": src_data["label"],
                        },
                        "target": {
                            "text": dst_data["text"],
                            "label": dst_data["label"],
                        },
                        "type": edge_data["rel"],
                    }
                )

        # Store in KuzuDB if available
        if self._conn:
            self._store_in_db(lex_graph)

        return {
            "entities": entities,
            "relations": relations,
        }

    def _store_in_db(self, lex_graph: nx.Graph) -> None:
        """Store lexical graph in KuzuDB."""
        if not self._conn:
            return

        # First pass: store all entities
        for node_id, node_data in lex_graph.nodes(data=True):
            if node_data.get("kind") == "Entity":
                try:
                    self._conn.execute(
                        """
                        MERGE (e:Entity {text: ?, label: ?})
                        """,
                        [node_data["text"], node_data["label"]],
                    )
                except Exception as e:
                    logging.error(f"Error storing entity: {e}")
                    pass

        # Second pass: store relationships
        for src, dst, edge_data in lex_graph.edges(data=True):
            src_data = lex_graph.nodes[src]
            dst_data = lex_graph.nodes[dst]

            if src_data.get("kind") == "Entity" and dst_data.get("kind") == "Entity":
                # Only store meaningful relationships (not CO_OCCURS_WITH)
                if edge_data["rel"] != "CO_OCCURS_WITH":
                    try:
                        self._conn.execute(
                            """
                            MATCH (e1:Entity {text: ?, label: ?}), 
                                  (e2:Entity {text: ?, label: ?})
                            CREATE (e1)-[r:Relation {type: ?, prob: ?}]->(e2)
                            """,
                            [
                                src_data["text"],
                                src_data["label"],
                                dst_data["text"],
                                dst_data["label"],
                                edge_data["rel"],
                                edge_data.get("prob", 1.0),
                            ],
                        )
                    except Exception as e:
                        logging.error(f"Error creating relationship: {e}")
                        pass

    def query(self, cypher_query: str) -> List[Dict]:
        """Execute a Cypher query against the knowledge graph.

        Args:
            cypher_query: The Cypher query to execute

        Returns:
            List of dictionaries containing query results
        """
        if not self._conn:
            return []

        try:
            result = self._conn.execute(cypher_query)
            results = []

            while result.has_next():
                row = result.get_next()
                # Convert row to dict and clean any None values
                row_dict = {
                    k: v for k, v in zip(result.column_names, row) if v is not None
                }
                if row_dict:  # Only add non-empty results
                    results.append(row_dict)

            return results
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            return []
