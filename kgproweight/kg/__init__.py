"""Knowledge Graph subsystem: entity linking, Wikidata SPARQL, embeddings."""

from kgproweight.kg.cache import EntityCache, SubgraphCache
from kgproweight.kg.coverage import coverage_score, graph_density, triple_in_subgraph
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.kg.kg_embeddings import KGEmbeddingModel, load_kg_embeddings
from kgproweight.kg.wikidata_retriever import WikidataSubgraphRetriever

__all__ = [
    "EntityLinker",
    "WikidataSubgraphRetriever",
    "EntityCache",
    "SubgraphCache",
    "coverage_score",
    "graph_density",
    "triple_in_subgraph",
    "KGEmbeddingModel",
    "load_kg_embeddings",
]
