"""Knowledge Graph module for entity and relationship extraction.

This implementation is inspired by and adapted from:
GraphGeeks.org talk 2024-08-14 https://live.zoho.com/PBOB6fvr6c
Original source: https://raw.githubusercontent.com/DerwenAI/strwythura/refs/heads/main/demo.py

The Knowledge Graph implementation uses:
- GLiNER for Named Entity Recognition
- GLiREL for Relationship Extraction
- KuzuDB for graph storage and querying
- spaCy for text processing and chunking
- TextRank for entity ranking

The graph is constructed in layers:
1. Base layer: Textual analysis using spaCy parse trees and noun chunks
2. Entity layer: Named entities from GLiNER overlaid on noun chunks
3. Relationship layer: Semantic relationships from GLiREL
4. TextRank layer: Entity ranking using TextRank
5. Storage layer: Persistent graph storage in KuzuDB

Note: Noun chunks are used to build the base lexical graph layer,
which helps provide context for entity and relationship extraction.
GLiNER entities are then overlaid on this base layer for more
accurate entity recognition and relationship extraction.
"""

from collections import defaultdict
from dataclasses import dataclass
import itertools
import logging
import math
import uuid
from icecream import ic
import os
import glirel
import unicodedata
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import spacy
from spacy.language import Language
import transformers
import networkx as nx
import kuzu

TR_ALPHA: float = 0.85
TR_LOOKBACK: int = 3

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
class TextChunk:
    id: uuid.UUID  # Which text chunk
    chunk_id: int  # Text chunk ID
    text: str  # Text of the chunk


@dataclass(order=False, frozen=False)
class Entity:
    """Entity class for storing extracted entities."""

    id: uuid.UUID  # Which text chunk
    loc: Tuple[int]  # Location in text
    key: str  # Unique identifier
    text: str  # Original text
    label: str  # Entity type
    chunk_id: int  # Text chunk ID
    sent_id: int  # Which sentence
    span: spacy.tokens.span.Span  # Original spaCy span
    node: Optional[int] = None  # Graph node ID


class KnowledgeGraph:
    """Knowledge graph with KuzuDB storage."""

    # Constants and configuration
    CHUNK_SIZE: int = 2048
    GLINER_MODEL: str = "urchade/gliner_multi-v2.1"  # Multi-language model
    SPACY_MODEL: str = "en_core_web_lg"  # Large web model

    # Entity labels we want to extract

    def __init__(
        self,
        db_path: Optional[str] = None,
        gliner_model: Optional[str] = None,
        spacy_model: Optional[str] = None,
    ):
        """Initialize KG with KuzuDB storage.

        Args:
            db_path: Path to KuzuDB database. If None, uses in-memory mode.
                    If ":memory:", uses in-memory mode explicitly.
                    Otherwise uses on-disk mode at the specified path.
            gliner_model: Optional GLiNER model name. Defaults to multi-language model.
            spacy_model: Optional spaCy model name. Defaults to large web model.
        """
        # Set model names
        self._gliner_model = gliner_model or self.GLINER_MODEL
        self._spacy_model = spacy_model or self.SPACY_MODEL

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
        nlp = spacy.load(self._spacy_model)

        # Add GLiNER
        nlp.add_pipe(
            "gliner_spacy",
            config={
                "gliner_model": self._gliner_model,
                "labels": NER_LABELS,
                "chunk_size": self.CHUNK_SIZE,
                "style": "ent",
            },
        )

        # Add GLiREL exactly like demo.py
        nlp.add_pipe("glirel", after="ner")

        return nlp

    def _parse_text(
        self,
        known_lemma: List[str],
        lex_graph: nx.Graph,
        chunk: TextChunk,
        *,
        debug: bool = False,
    ) -> spacy.tokens.doc.Doc:
        """
        Parse an input text chunk, returning a `spaCy` document.
        """
        doc: spacy.tokens.doc.Doc = list(
            self._nlp.pipe(
                [(chunk.text, RE_LABELS)],
                as_tuples=True,
            )
        )[0][0]

        # scan the document tokens to add lemmas to _lexical graph_ using
        # a _textgraph_ approach called the _textrank_ algorithm
        for sent in doc.sents:
            node_seq: List[int] = []

            if debug:
                ic(sent)

            for tok in sent:
                text: str = tok.text.strip()

                if tok.pos_ in ["NOUN", "PROPN"]:
                    key: str = tok.pos_ + "." + tok.lemma_.strip().lower()
                    prev_known: bool = False

                    if key not in known_lemma:
                        # create a new node
                        known_lemma.append(key)
                    else:
                        # link to an existing node, adding weight
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
                            chunk=chunk,
                            count=1,
                        )

                    elif prev_known:
                        node: dict = lex_graph.nodes[node_id]
                        node["count"] += 1

            # create the _textrank_ edges for the lexical graph,
            # which will get used for ranking, but discarded later
            if debug:
                ic(node_seq)

            for hop in range(TR_LOOKBACK):
                for node_id, node in enumerate(node_seq[: -1 - hop]):
                    neighbor: int = node_seq[hop + node_id + 1]

                    if not lex_graph.has_edge(node, neighbor):
                        lex_graph.add_edge(
                            node,
                            neighbor,
                            rel="FOLLOWS_LEXICALLY",
                        )

        return doc

    def _init_db(self) -> None:
        """Initialize KuzuDB schema.

        Creates the following tables:
        - Entity (node table): Stores entities with text and label
        - Relation (relationship table): Stores relationships between entities
        """
        if not self._conn:
            return

        try:
            # Drop existing tables if they exist
            try:
                self._conn.execute("DROP TABLE IF EXISTS Relation")
                self._conn.execute("DROP TABLE IF EXISTS Entity")
            except Exception as e:
                logging.debug(f"Tables don't exist yet: {e}")

            # Create Entity node table with proper schema
            # STRING is variable-length character string with UTF-8 encoding
            self._conn.execute(
                """
                CREATE NODE TABLE Entity (
                    id SERIAL,
                    text STRING,
                    label STRING,
                    PRIMARY KEY (id)
                )
                """
            )
            logging.info("Created Entity table")

            # Create relationship table with proper schema
            # DOUBLE is 8-byte double precision floating point
            self._conn.execute(
                """
                CREATE REL TABLE Relation (
                    FROM Entity TO Entity,
                    type STRING,
                    prob DOUBLE DEFAULT 1.0
                )
                """
            )
            logging.info("Created Relation table")

        except Exception as e:
            logging.error(f"Error initializing database: {e}")
            raise  # Re-raise to ensure we know if initialization failed

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
        self, id: uuid.UUID, doc: spacy.tokens.doc.Doc
    ) -> Sequence[TextChunk]:
        """Split document into text chunks."""
        chunks: List[str] = []
        chunk_total: int = 0
        prev_line: str = ""
        chunk_list: List[TextChunk] = []
        chunk_id: int = 0

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
        chunk_list.append(TextChunk(id=id, chunk_id=chunk_id, text="\n".join(chunks)))
        return chunk_list

    def _make_entity(
        self,
        span_decoder: Dict[tuple, Entity],
        sent_map: Dict[spacy.tokens.span.Span, int],
        span: spacy.tokens.span.Span,
        chunk: TextChunk,
        *,
        debug: bool = False,
    ) -> None:
        """Create an Entity from a spaCy span.

        This handles both named entities from GLiNER and noun chunks from spaCy.
        Noun chunks form the base layer of the lexical graph, while GLiNER entities
        are overlaid on top for more accurate entity recognition.
        """
        sent = span.sent
        sent_id = sent_map.get(sent, -1)

        # For noun chunks, use POS tags to build compound key
        if not hasattr(span, "label_"):
            key = " ".join(
                [tok.pos_ + "." + tok.lemma_.strip().lower() for tok in span]
            )
            label = "NP"  # Noun Phrase
        else:
            # For GLiNER entities, use the entity label
            key = f"{span.label_}.{span.text}"
            label = span.label_

        ent = Entity(
            id=chunk.id,
            loc=(span.start, span.end),
            key=key,
            text=span.text,
            label=label,
            chunk_id=chunk.chunk_id,
            sent_id=sent_id,
            span=span,
        )

        # Only store if not already seen or if it's a GLiNER entity
        if ent.loc not in span_decoder or hasattr(span, "label_"):
            span_decoder[ent.loc] = ent
            if debug:
                ic(ent)

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

    def _abstract_overlay(
        self,
        chunk_list: List[TextChunk],
        lex_graph: nx.Graph,
        sem_overlay: nx.Graph,
    ) -> None:
        """
        Abstract a _semantic overlay_ from the lexical graph -- in other words
        which nodes and edges get promoted up to the next level?

        Also connect the extracted entities with their source chunks, where
        the latter first-class citizens within the KG.
        """
        kept_nodes: Set[int] = set()
        skipped_rel: Set[str] = set(["FOLLOWS_LEXICALLY", "COMPOUND_ELEMENT_OF"])

        chunk_nodes: Dict[int, str] = {
            chunk.uid: f"chunk_{chunk.uid}" for chunk in chunk_list
        }

        for chunk_id, node_id in chunk_nodes.items():
            sem_overlay.add_node(
                node_id,
                kind="Chunk",
                chunk=chunk_id,
            )

        for node_id, node_attr in lex_graph.nodes(data=True):
            if node_attr["kind"] == "Entity":
                kept_nodes.add(node_id)
                count: int = node_attr["count"]

                if not sem_overlay.has_node(node_id):
                    sem_overlay.add_node(
                        node_id,
                        kind="Entity",
                        key=node_attr["key"],
                        text=node_attr["text"],
                        label=node_attr["label"],
                        count=count,
                    )
                else:
                    sem_overlay.nodes[node_id]["count"] += count

                sem_overlay.add_edge(
                    node_id,
                    chunk_nodes[node_attr["chunk"]],
                    rel="WITHIN",
                    weight=node_attr["rank"],
                )

        for src_id, dst_id, edge_attr in lex_graph.edges(data=True):
            if src_id in kept_nodes and dst_id in kept_nodes:
                rel: str = edge_attr["rel"]
                prob: float = 1.0

                if "prob" in edge_attr:
                    prob = edge_attr["prob"]

                if rel not in skipped_rel:
                    if not sem_overlay.has_edge(src_id, dst_id):
                        sem_overlay.add_edge(
                            src_id,
                            dst_id,
                            rel=rel,
                            prob=prob,
                        )
                    else:
                        sem_overlay[src_id][dst_id]["prob"] = max(
                            prob,
                            sem_overlay.edges[(src_id, dst_id)]["prob"],
                        )

    def _run_textrank(
        self,
        lex_graph: nx.Graph,
    ) -> pd.DataFrame:
        """
        Run eigenvalue centrality (i.e., _Personalized PageRank_) to rank the entities.
        """
        # build a dataframe of node ranks and counts
        df_rank: pd.DataFrame = pd.DataFrame.from_dict(
            [
                {
                    "node_id": node,
                    "weight": rank,
                    "count": lex_graph.nodes[node]["count"],
                }
                for node, rank in nx.pagerank(
                    lex_graph, alpha=TR_ALPHA, weight="count"
                ).items()
            ]
        )

        # normalize by column and calculate quantiles
        df1: pd.DataFrame = df_rank[["count", "weight"]].apply(
            lambda x: x / x.max(), axis=0
        )
        bins: np.ndarray = calc_quantile_bins(len(df1.index))

        # stripe each columns
        df2: pd.DataFrame = pd.DataFrame(
            [stripe_column(values, bins) for _, values in df1.items()]
        ).T

        # renormalize the ranks
        df_rank["rank"] = df2.apply(root_mean_square, axis=1)
        rank_col: np.ndarray = df_rank["rank"].to_numpy()
        rank_col /= sum(rank_col)
        df_rank["rank"] = rank_col

        # move the ranked weights back into the graph
        for _, row in df_rank.iterrows():
            node: int = row["node_id"]
            lex_graph.nodes[node]["rank"] = row["rank"]

        df: pd.DataFrame = pd.DataFrame(
            [
                node_attr
                for node, node_attr in lex_graph.nodes(data=True)
                if node_attr["kind"] == "Entity"
            ]
        ).sort_values(by=["rank"], ascending=False)

        return df

    def analyze(self, id: uuid.UUID, text: str, *, debug: bool = False) -> Dict:
        """Analyze text to extract entities and relationships."""
        # Initialize processing structures
        chunk_list: List[TextChunk] = []
        lex_graph: nx.Graph = nx.Graph()
        sem_overlay: nx.Graph = nx.Graph()
        known_lemma: List[str] = []

        # Split text into chunks and process with GLiREL labels
        doc: spacy.tokens.doc.Doc = list(
            self._nlp.pipe(
                [(text, RE_LABELS)],
                as_tuples=True,
            )
        )[0][0]

        chunk_list = self._make_chunk(id, doc)

        # Process each chunk
        for chunk in chunk_list:
            span_decoder: Dict[tuple, Entity] = {}
            # Parse the chunk with GLiREL labels
            doc = self._parse_text(known_lemma, lex_graph, chunk, debug=debug)
            # keep track of sentence numbers per chunk, to use later
            # for entity co-occurrence links
            sent_map: Dict[spacy.tokens.span.Span, int] = {}

            # Scan document tokens to add lemmas to lexical graph using textrank
            for sent_id, sent in enumerate(doc.sents):
                sent_map[sent] = sent_id

                if debug:
                    ic(f"Processing sentence: {sent}")

            # First process noun chunks to build base lexical layer
            for span in doc.noun_chunks:
                if debug:
                    ic(f"Processing noun chunk: {span}")
                self._make_entity(span_decoder, sent_map, span, chunk, debug=debug)

            # Then overlay GLiNER entities
            for span in doc.ents:
                if debug:
                    ic(f"Processing named entity: {span}")
                self._make_entity(span_decoder, sent_map, span, chunk, debug=debug)

            # Extract entities and add to lexical graph
            for ent in span_decoder.values():
                if ent.key not in STOP_WORDS:
                    self._extract_entity(
                        known_lemma,
                        lex_graph,
                        ent,
                        debug=debug,
                    )

            # Extract relations for co-occurring entity pairs
            self._extract_relations(
                known_lemma,
                lex_graph,
                span_decoder,
                sent_map,
                doc,
                chunk,
                debug=debug,
            )

            # Connect entities which co-occur within the same sentence
            self._connect_entities(
                lex_graph,
                span_decoder,
            )

        df: pd.DataFrame = self._run_textrank(
            lex_graph,
        )

        ic(df.head(20))

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
            "textrank": df.to_dict(orient="records"),
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
                        MERGE (e:Entity {text: $text, label: $label})
                        """,
                        {"text": node_data["text"], "label": node_data["label"]},
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
                            MATCH (e1:Entity {text: $src_text, label: $src_label}), 
                                  (e2:Entity {text: $dst_text, label: $dst_label})
                            CREATE (e1)-[r:Relation {type: $rel_type, prob: $prob}]->(e2)
                            """,
                            {
                                "src_text": src_data["text"],
                                "src_label": src_data["label"],
                                "dst_text": dst_data["text"],
                                "dst_label": dst_data["label"],
                                "rel_type": edge_data["rel"],
                                "prob": edge_data.get("prob", 1.0),
                            },
                        )
                    except Exception as e:
                        logging.error(f"Error creating relationship: {e}")
                        pass

    def query(self, cypher_query: str, params: Dict = {}) -> List[Dict]:
        """Execute a Cypher query against the knowledge graph.

        Args:
            cypher_query: The Cypher query to execute

        Returns:
            List of dictionaries containing query results
        """
        if not self._conn:
            return []

        try:
            if params:
                result = self._conn.execute(cypher_query, params)
            else:
                result = self._conn.execute(cypher_query)

            results = []

            while result.has_next():
                row = result.get_next()
                # Convert row to dict and clean any None values
                row_dict = {
                    k: v
                    for k, v in zip(result.get_column_names(), row)
                    if v is not None
                }
                if row_dict:  # Only add non-empty results
                    results.append(row_dict)

            return results
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            return []


######################################################################
## numerical utilities


def calc_quantile_bins(
    num_rows: int,
    *,
    amplitude: int = 4,
) -> np.ndarray:
    """
    Calculate the bins to use for a quantile stripe,
    using [`numpy.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)

        num_rows:
    number of rows in the target dataframe

        returns:
    calculated bins, as a `numpy.ndarray`
    """
    granularity = max(round(math.log(num_rows) * amplitude), 1)

    return np.linspace(
        0,
        1,
        num=granularity,
        endpoint=True,
    )


def stripe_column(
    values: list,
    bins: int,
) -> np.ndarray:
    """
    Stripe a column in a dataframe, by interpolating quantiles into a set of discrete indexes.

        values:
    list of values to stripe

        bins:
    quantile bins; see [`calc_quantile_bins()`](#calc_quantile_bins-function)

        returns:
    the striped column values, as a `numpy.ndarray`
    """
    s = pd.Series(values)
    q = s.quantile(bins, interpolation="nearest")

    try:
        stripe = np.digitize(values, q) - 1
        return stripe
    except ValueError as ex:
        # should never happen?
        print("ValueError:", str(ex), values, s, q, bins)
        raise


def root_mean_square(values: List[float]) -> float:
    """
    Calculate the [*root mean square*](https://mathworld.wolfram.com/Root-Mean-Square.html)
    of the values in the given list.

        values:
    list of values to use in the RMS calculation

        returns:
    RMS metric as a float
    """
    s: float = sum(map(lambda x: float(x) ** 2.0, values))
    n: float = float(len(values))

    return math.sqrt(s / n)
