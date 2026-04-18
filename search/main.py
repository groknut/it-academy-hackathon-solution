import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any
import time
from datetime import datetime

import httpx
from fastembed import SparseTextEmbedding
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import Filter, FieldCondition, Range  # <-- добавлено

EMBEDDINGS_DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Ваш сервис должен считывать эти переменные из окружения (env), так как проверяющая система управляет ими
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8003"))

API_KEY = os.getenv("API_KEY")
EMBEDDINGS_DENSE_URL = os.getenv("EMBEDDINGS_DENSE_URL")
QDRANT_DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
QDRANT_SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANKER_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
RERANKER_URL = os.getenv("RERANKER_URL")
OPEN_API_LOGIN = os.getenv("OPEN_API_LOGIN")
OPEN_API_PASSWORD = os.getenv("OPEN_API_PASSWORD")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "evaluation")
REQUIRED_ENV_VARS = [
    "EMBEDDINGS_DENSE_URL",
    "RERANKER_URL",
    "QDRANT_URL",
]

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("search-service")


def validate_required_env() -> None:
    if bool(OPEN_API_LOGIN) != bool(OPEN_API_PASSWORD):
        raise RuntimeError("OPEN_API_LOGIN and OPEN_API_PASSWORD must be set together")

    if not API_KEY and not (OPEN_API_LOGIN and OPEN_API_PASSWORD):
        raise RuntimeError("Either API_KEY or OPEN_API_LOGIN and OPEN_API_PASSWORD must be set")

    missing_env_vars = [
        name for name in REQUIRED_ENV_VARS if os.getenv(name) is None or os.getenv(name) == ""
    ]
    if not missing_env_vars:
        return

    logger.error("Empty required env vars: %s", ", ".join(missing_env_vars))
    raise RuntimeError(f"Empty required env vars: {', '.join(missing_env_vars)}")


validate_required_env()


def get_upstream_request_kwargs() -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    kwargs: dict[str, Any] = {"headers": headers}

    if OPEN_API_LOGIN and OPEN_API_PASSWORD:
        kwargs["auth"] = (OPEN_API_LOGIN, OPEN_API_PASSWORD)
        return kwargs

    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    return kwargs


# Модель данных, которую мы предоставляем и рассчитываем получать от вас
class DateRange(BaseModel):
    from_: str = Field(alias="from")
    to: str


class Entities(BaseModel):
    people: list[str] | None = None
    emails: list[str] | None = None
    documents: list[str] | None = None
    names: list[str] | None = None
    links: list[str] | None = None


class Question(BaseModel):
    text: str
    asker: str = ""
    asked_on: str = ""
    variants: list[str] | None = None
    hyde: list[str] | None = None
    keywords: list[str] | None = None
    entities: Entities | None = None
    date_mentions: list[str] | None = None
    date_range: DateRange | None = None
    search_text: str = ""


class SearchAPIRequest(BaseModel):
    question: Question


class SearchAPIItem(BaseModel):
    message_ids: list[str]


class SearchAPIResponse(BaseModel):
    results: list[SearchAPIItem]


class DenseEmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class DenseEmbeddingResponse(BaseModel):
    data: list[DenseEmbeddingItem]


class SparseVector(BaseModel):
    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)


class SparseEmbeddingResponse(BaseModel):
    vectors: list[SparseVector]

# Метадата чанков в Qdrant'e, по которой вы можете фильтровать
class ChunkMetadata(BaseModel):
    chat_name: str
    chat_type: str  # channel, group, private, thread
    chat_id: str
    chat_sn: str
    thread_sn: str | None = None
    message_ids: list[str]
    start: str
    end: str
    participants: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    contains_forward: bool = False
    contains_quote: bool = False


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    logger.info("Loading local sparse model %s", SPARSE_MODEL_NAME)
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient()
    app.state.qdrant = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=API_KEY,
    )
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()


app = FastAPI(title="Search Service", version="0.1.0", lifespan=lifespan)


# Внутри шаблона dense и rerank берутся из внешних HTTP endpoint'ов,
# которые предоставляет проверяющая система.
# Текущий код ниже — минимальный пример search pipeline.
DENSE_PREFETCH_K = 10
SPRASE_PREFETCH_K = 30
RETRIEVE_K = 20
RERANK_LIMIT = 10


# ---------- Функции для работы с метаданными ----------
def build_date_filter(date_range: DateRange | None) -> Filter | None:
    """Создаёт фильтр Qdrant по диапазону дат (используя поля metadata.start и metadata.end)."""
    if not date_range:
        return None
    return Filter(
        must=[
            FieldCondition(
                key="metadata.start",
                range=Range(gte=date_range.from_),
            ),
            FieldCondition(
                key="metadata.end",
                range=Range(lte=date_range.to),
            ),
        ]
    )


def build_metadata_filter(question: Question) -> Filter | None:
    """Создаёт фильтр по типу чата, участникам и т.д."""
    conditions = []

    # Фильтр по типу чата (если есть подсказки в тексте)
    text_lower = question.text.lower()
    if "личк" in text_lower or "private" in text_lower:
        conditions.append(FieldCondition(key="metadata.chat_type", match={"value": "private"}))
    elif "канал" in text_lower or "channel" in text_lower:
        conditions.append(FieldCondition(key="metadata.chat_type", match={"value": "channel"}))
    elif "групп" in text_lower or "group" in text_lower:
        conditions.append(FieldCondition(key="metadata.chat_type", match={"value": "group"}))

    # Фильтр по участникам (если в entities.people есть список)
    if question.entities and question.entities.people:
        conditions.append(
            FieldCondition(
                key="metadata.participants",
                match={"any": question.entities.people},
            )
        )

    if not conditions:
        return None
    return Filter(must=conditions)


def combine_filters(f1: Filter | None, f2: Filter | None) -> Filter | None:
    """Объединяет два фильтра через must."""
    if f1 is None and f2 is None:
        return None
    if f1 is None:
        return f2
    if f2 is None:
        return f1
    # Объединяем условия
    must = []
    if f1.must:
        must.extend(f1.must)
    if f2.must:
        must.extend(f2.must)
    return Filter(must=must)


def time_decay_score(point: Any, current_time: float | None = None) -> float:
    """Возвращает множитель (0.1..1) на основе давности последнего сообщения в чанке."""
    if current_time is None:
        current_time = time.time()
    metadata = point.payload.get("metadata", {})
    end_str = metadata.get("end")
    if not end_str:
        return 1.0
    try:
        end_ts = datetime.fromisoformat(end_str).timestamp()
        age_days = (current_time - end_ts) / (24 * 3600)
        # экспоненциальное затухание: период полураспада 30 дней
        decay = 0.5 ** (age_days / 30)
        return max(decay, 0.1)
    except Exception:
        return 1.0


def boost_by_metadata(point: Any, question: Question) -> float:
    """Дополнительные бусты на основе метаданных."""
    boost = 1.0
    metadata = point.payload.get("metadata", {})
    # Буст за пересланные сообщения
    if metadata.get("contains_forward"):
        boost *= 1.2
    # Буст, если в чанке упоминается тот, кто задаёт вопрос
    if question.asker and question.asker in metadata.get("mentions", []):
        boost *= 1.5
    return boost


# ---------- Основные функции поиска ----------
async def embed_dense(client: httpx.AsyncClient, text: str) -> list[float]:
    # Dense endpoint ожидает OpenAI-compatible body с input как списком строк.
    response = await client.post(
        EMBEDDINGS_DENSE_URL,
        **get_upstream_request_kwargs(),
        json={
            "model": os.getenv("EMBEDDINGS_DENSE_MODEL", EMBEDDINGS_DENSE_MODEL),
            "input": [text],
        },
    )
    response.raise_for_status()

    payload = DenseEmbeddingResponse.model_validate(response.json())
    if not payload.data:
        raise ValueError("Dense embedding response is empty")

    return payload.data[0].embedding


async def embed_sparse(text: str) -> SparseVector:
    vectors = list(get_sparse_model().embed([text]))
    if not vectors:
        raise ValueError("Sparse embedding response is empty")

    item = vectors[0]
    return SparseVector(
        indices=[int(index) for index in item.indices.tolist()],
        values=[float(value) for value in item.values.tolist()],
    )


async def qdrant_search(
    client: AsyncQdrantClient,
    dense_vector: list[float],
    sparse_vector: SparseVector,
    query_filter: Filter | None = None,  # <-- добавлен фильтр
) -> Any | None:
    response = await client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using=QDRANT_DENSE_VECTOR_NAME,
                limit=DENSE_PREFETCH_K,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values,
                ),
                using=QDRANT_SPARSE_VECTOR_NAME,
                limit=SPRASE_PREFETCH_K,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=RETRIEVE_K,
        with_payload=True,
        filter=query_filter,  # <-- применяем фильтр
    )

    if not response.points:
        return None

    return response.points


def extract_message_ids(point: Any) -> list[str]:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    message_ids = metadata.get("message_ids") or []

    return [str(message_id) for message_id in message_ids]


async def get_rerank_scores(
    client: httpx.AsyncClient,
    label: str,
    targets: list[str],
) -> list[float]:
    if not targets:
        return []

    # Rerank endpoint возвращает score для пары query -> candidate text.
    response = await client.post(
        RERANKER_URL,
        **get_upstream_request_kwargs(),
        json={
            "model": RERANKER_MODEL,
            "encoding_format": "float",
            "text_1": label,
            "text_2": targets,
        },
    )
    response.raise_for_status()

    payload = response.json()
    data = payload.get("data") or []

    return [float(sample["score"]) for sample in data]


async def rerank_points(
    client: httpx.AsyncClient,
    query: str,
    points: list[Any],
) -> list[Any]:
    rerank_candidates = points[:10]
    rerank_targets = [point.payload.get("page_content") for point in rerank_candidates]
    scores = await get_rerank_scores(client, query, rerank_targets)

    reranked_candidates = [
        point
        for _, point in sorted(
            zip(scores, rerank_candidates, strict=True),
            key=lambda item: item[0],
            reverse=True,
        )
    ]

    return reranked_candidates


# Ваш сервис должен имплементировать оба этих метода
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchAPIResponse)
async def search(payload: SearchAPIRequest) -> SearchAPIResponse:
    # Используем search_text, если он предоставлен
    query = payload.question.search_text.strip() or payload.question.text.strip()
    if not query:
        raise HTTPException(status_code=400, detail="question.text or question.search_text is required")

    client: httpx.AsyncClient = app.state.http
    qdrant: AsyncQdrantClient = app.state.qdrant

    # 1. Строим фильтры по метаданным
    date_filter = build_date_filter(payload.question.date_range)
    metadata_filter = build_metadata_filter(payload.question)
    combined_filter = combine_filters(date_filter, metadata_filter)

    # 2. Получаем векторы
    dense_vector = await embed_dense(client, query)
    sparse_vector = await embed_sparse(query)

    # 3. Поиск с фильтром
    points = await qdrant_search(qdrant, dense_vector, sparse_vector, combined_filter)
    if not points:
        return SearchAPIResponse(results=[])

    # 4. Пост-обработка: временной decay и бустинг по метаданным
    for point in points:
        point.score = point.score * time_decay_score(point) * boost_by_metadata(point, payload.question)

    # 5. Сортировка по новому скору
    points = sorted(points, key=lambda p: p.score, reverse=True)

    # 6. Реранк (уже на меньшем количестве кандидатов)
    top_for_rerank = points[:RERANK_LIMIT]
    reranked = await rerank_points(client, query, top_for_rerank)

    # 7. Сбор message_ids
    message_ids = []
    for point in reranked:
        message_ids.extend(extract_message_ids(point))

    # Убираем дубликаты, сохраняя порядок
    seen = set()
    unique_ids = []
    for mid in message_ids:
        if mid not in seen:
            seen.add(mid)
            unique_ids.append(mid)

    return SearchAPIResponse(results=[SearchAPIItem(message_ids=unique_ids)])


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)
    detail = str(exc) or repr(exc)

    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return JSONResponse(status_code=500, content={"detail": detail})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()