import logging
import os
from functools import lru_cache
from typing import Any
import asyncio
import hashlib
from datetime import datetime  # <-- добавлено

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Ваш сервис должен считывать эти переменные из окружения (env), так как проверяющая система управляет ими
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8004"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("index-service")


# Модель данных, которую мы предоставляем и рассчитываем получать от вас
class Chat(BaseModel):
    id: str
    name: str
    sn: str
    type: str  # group, channel, private
    is_public: bool | None = None
    members_count: int | None = None
    members: list[dict[str, Any]] | None = None


class Message(BaseModel):
    id: str
    thread_sn: str | None = None
    time: int
    text: str
    sender_id: str
    file_snippets: str
    parts: list[dict[str, Any]] | None = None
    mentions: list[str] | None = None
    member_event: dict[str, Any] | None = None
    is_system: bool
    is_hidden: bool
    is_forward: bool
    is_quote: bool


class ChatData(BaseModel):
    chat: Chat
    overlap_messages: list[Message]
    new_messages: list[Message]


class IndexAPIRequest(BaseModel):
    data: ChatData


# dense_content будет передан в dense embedding модель для построения семантического вектора.
# sparse_content будет передан в sparse модель для построения разреженного индекса "по словам".
# Можно оставить dense_content и sparse_content равными page_content,
# а можно формировать для них разные версии текста.
class IndexAPIItem(BaseModel):
    page_content: str
    dense_content: str
    sparse_content: str
    message_ids: list[str]
    metadata: dict[str, Any]  # <-- добавлено


class IndexAPIResponse(BaseModel):
    results: list[IndexAPIItem]


class SparseEmbeddingRequest(BaseModel):
    texts: list[str]


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]


class SparseEmbeddingResponse(BaseModel):
    vectors: list[SparseVector]


app = FastAPI(title="Index Service", version="0.1.0")

# Ваша внутренняя логика построения чанков. Можете делать всё, что посчитаете нужным.
# Текущий код – минимальный пример

CHUNK_SIZE = 512
OVERLAP_SIZE = 256
SPARSE_MODEL_NAME = "Qdrant/bm25"
FASTEMBED_CACHE_PATH = "/models/fastembed"

# Важная переманная, которая позволяет вычислять sparse вектор в несколько ядер. Не рекомендуется изменять.
UVICORN_WORKERS = 8


def render_message(message: Message) -> str:
    text = ""

    if message.text:
        text += message.text

    if message.parts:
        parts_text: list[str] = []
        for part in message.parts:
            # parts различаются по своему типу, см. README.md
            part_text = part.get("text")
            if isinstance(part_text, str) and part_text:
                parts_text.append(part_text)
        if parts_text:
            text += "\n".join(parts_text)

    return text


def build_chunks(
    chat: Chat,  # <-- добавлен параметр
    overlap_messages: list[Message],
    new_messages: list[Message],
) -> list[IndexAPIItem]:
    result: list[IndexAPIItem] = []

    def build_text_and_ranges(messages: list[Message]) -> tuple[str, list[tuple[int, int, str]]]:
        text_parts: list[str] = []
        message_ranges: list[tuple[int, int, str]] = []
        position = 0

        for index, message in enumerate(messages):
            text = render_message(message)
            if not text:
                continue

            if index > 0 and text_parts:
                text_parts.append("\n")
                position += 1

            start = position
            text_parts.append(text)
            position += len(text)
            message_ranges.append((start, position, message.id))

        return "".join(text_parts), message_ranges

    def slice_tail(
        text: str,
        tail_size: int,
    ) -> str:
        if tail_size <= 0:
            return ""

        tail_start = max(0, len(text) - tail_size)
        return text[tail_start:]

    overlap_text, overlap_message_ranges = build_text_and_ranges(overlap_messages)
    previous_chunk_text = slice_tail(overlap_text, OVERLAP_SIZE)

    new_text, new_message_ranges = build_text_and_ranges(new_messages)

    # Для быстрого доступа к сообщениям по id
    new_messages_by_id = {msg.id: msg for msg in new_messages}
    overlap_messages_by_id = {msg.id: msg for msg in overlap_messages}

    for start in range(0, len(new_text), CHUNK_SIZE):
        chunk_body = new_text[start: start + CHUNK_SIZE]
        if not chunk_body:
            continue

        chunk_body_ranges = [
            (
                max(message_start, start) - start,
                min(message_end, start + len(chunk_body)) - start,
                message_id,
            )
            for message_start, message_end, message_id in new_message_ranges
            if message_end > start and message_start < start + len(chunk_body)
        ]
        chunk_overlap = previous_chunk_text
        chunk_text = chunk_overlap
        if chunk_text and chunk_body:
            chunk_text += "\n"
        chunk_text += chunk_body

        # Собираем ID сообщений в чанке
        msg_ids_in_chunk = [msg_id for _, _, msg_id in chunk_body_ranges]

        # --- Формирование метаданных ---
        # Даты начала и конца
        times = []
        for msg_id in msg_ids_in_chunk:
            msg = new_messages_by_id.get(msg_id) or overlap_messages_by_id.get(msg_id)
            if msg:
                times.append(msg.time)
        start_dt = datetime.fromtimestamp(min(times)).isoformat() if times else ""
        end_dt = datetime.fromtimestamp(max(times)).isoformat() if times else ""

        # Участники (из чата)
        participants = []
        if chat.members:
            participants = [str(m.get("id")) for m in chat.members if m.get("id")]

        # Упоминания (агрегируем из сообщений чанка)
        mentions_set = set()
        for msg_id in msg_ids_in_chunk:
            msg = new_messages_by_id.get(msg_id) or overlap_messages_by_id.get(msg_id)
            if msg and msg.mentions:
                mentions_set.update(msg.mentions)

        # Флаги пересылки и цитирования
        contains_forward = any(
            (new_messages_by_id.get(mid) and new_messages_by_id[mid].is_forward)
            or (overlap_messages_by_id.get(mid) and overlap_messages_by_id[mid].is_forward)
            for mid in msg_ids_in_chunk
        )
        contains_quote = any(
            (new_messages_by_id.get(mid) and new_messages_by_id[mid].is_quote)
            or (overlap_messages_by_id.get(mid) and overlap_messages_by_id[mid].is_quote)
            for mid in msg_ids_in_chunk
        )

        # thread_sn – берём из первого попавшегося сообщения, у которого он есть
        thread_sn = None
        for msg_id in msg_ids_in_chunk:
            msg = new_messages_by_id.get(msg_id) or overlap_messages_by_id.get(msg_id)
            if msg and msg.thread_sn:
                thread_sn = msg.thread_sn
                break

        metadata = {
            "chat_name": chat.name,
            "chat_type": chat.type,
            "chat_id": chat.id,
            "chat_sn": chat.sn,
            "thread_sn": thread_sn,
            "start": start_dt,
            "end": end_dt,
            "participants": participants,
            "mentions": list(mentions_set),
            "contains_forward": contains_forward,
            "contains_quote": contains_quote,
        }

        result.append(
            IndexAPIItem(
                page_content=chunk_text,
                dense_content=chunk_text,
                sparse_content=chunk_text,
                message_ids=msg_ids_in_chunk,
                metadata=metadata,
            )
        )
        previous_chunk_text = slice_tail(chunk_text, OVERLAP_SIZE)

    return result


# Ваш сервис должен имплементировать оба этих метода
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/index", response_model=IndexAPIResponse)
async def index(payload: IndexAPIRequest) -> IndexAPIResponse:
    return IndexAPIResponse(
        results=build_chunks(
            payload.data.chat,                # <-- передаём chat
            payload.data.overlap_messages,
            payload.data.new_messages,
        )
    )


@lru_cache(maxsize=1)
def get_sparse_model():
    from fastembed import SparseTextEmbedding

    # можете делать любой вектор, который будет совместим с вашим поиском в Qdrant
    # помните об ограничении времени выполнения вашей работы в тестирующей системе
    logger.info(
        "Loading sparse model %s from cache %s",
        SPARSE_MODEL_NAME,
        FASTEMBED_CACHE_PATH,
    )
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def embed_sparse_texts(texts: list[str]) -> list[SparseVector]:
    model = get_sparse_model()
    vectors: list[dict[str, list[int] | list[float]]] = []

    for item in model.embed(texts):
        vectors.append(
            {
                "indices": item.indices.tolist(),
                "values": item.values.tolist(),
            }
        )

    return vectors


@app.post("/sparse_embedding")
async def sparse_embedding(payload: SparseEmbeddingRequest) -> dict[str, Any]:
    # Проверяющая система вызывает этот endpoint при создании коллекции
    vectors = await asyncio.to_thread(embed_sparse_texts, payload.texts)
    return {"vectors": vectors}


# красивая обработка ошибок
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)

    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    return JSONResponse(status_code=500, content={"detail": str(exc)})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        workers=UVICORN_WORKERS,
    )


if __name__ == "__main__":
    main()