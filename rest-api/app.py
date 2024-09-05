from litestar import Litestar, post
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()

@post("/embedding")
def embedding(data: dict[str, str]) -> dict[str, list]:
    """Embedding doc."""
    if 'doc' in data:
        doc_embeddings = model.encode(data['doc'])
        return {"embedding": doc_embeddings.tolist()}
    else:
        return {"embedding": []}

@post("/encode")
def encode(data: dict[str, str]) -> dict[str, list]:
    """Encode query"""
    if 'query' in data:
        query_embeddings = model.encode(data['query'], prompt_name='s2p_query')
        return {"embedding": query_embeddings.tolist()}
    else:
        return {"embedding": []}


app = Litestar(route_handlers=[embedding, encode])