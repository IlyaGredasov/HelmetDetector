import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List
from typing import Literal
from typing import Optional

import psycopg
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import Query
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from config import cfg

load_dotenv()

DB_CONFIG = (
    f"host={cfg.DB_HOST} "
    f"port={cfg.DB_PORT} "
    f"dbname={cfg.DB_NAME} "
    f"user={cfg.DB_USER} "
    f"password={cfg.DB_PASSWORD}"
)

pool: AsyncConnectionPool | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Инициализирует пул соединений к базе данных.

    :param _app: экземпляр FastAPI
    :return: контекстный менеджер lifespan
    """
    global pool
    pool = AsyncConnectionPool(
        conninfo=DB_CONFIG,
        min_size=cfg.DB_POOL_MIN,
        max_size=cfg.DB_POOL_MAX,
        open=False,
        kwargs={"autocommit": True},
    )
    await pool.open()
    try:
        yield
    finally:
        await pool.close()


helmet_db_api = FastAPI(lifespan=lifespan)


class DetectionStatus(BaseModel):
    """
    Модель статуса детекции.

    :param status: строковый статус
    """
    status: Literal["pending", "confirmed", "rejected"]


class Detection(DetectionStatus):
    """
    Модель детекции.

    :param detection_id: идентификатор детекции
    :param camera_id: идентификатор камеры
    :param detection_time: время фиксации
    """
    detection_id: int
    camera_id: int
    detection_time: datetime


@helmet_db_api.post("/detections", response_model=Detection)
async def create_detection(
        camera_id: int = Form(...),
        detection_time: datetime = Form(...),
        image: UploadFile = File(...)
):
    """
    Создаёт новую детекцию.

    :param camera_id: ID камеры
    :param detection_time: время фиксации
    :param image: изображение нарушения
    :return: созданная детекция
    """
    data = await image.read()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM cameras WHERE camera_id = %s",
                (camera_id,),
            )
            row = await cur.fetchone()
            if not row:
                raise HTTPException(status_code=400, detail="Unknown camera_id")

            await cur.execute(
                """
                INSERT INTO detections (camera_id, detection_time, image)
                VALUES (%s, %s, %s) RETURNING detection_id, camera_id, detection_time, status
                """,
                (camera_id, detection_time, psycopg.Binary(data)),
            )
            row = await cur.fetchone()
            return Detection(detection_id=row[0], camera_id=row[1], detection_time=row[2], status=row[3])


@helmet_db_api.get("/detections", response_model=List[Detection])
async def list_detections(
        status: Optional[Literal["pending", "confirmed", "rejected"]] = Query(None),
        camera_id: Optional[int] = Query(None),
        time_from: Optional[datetime] = Query(None),
        time_to: Optional[datetime] = Query(None),
        limit: int = Query(50, ge=1),
        order: Literal["asc", "desc"] = Query("desc")
):
    """
    Возвращает список детекций.

    :param status: фильтр по статусу
    :param camera_id: фильтр по ID камеры
    :param time_from: начальное время
    :param time_to: конечное время
    :param limit: ограничение количества
    :param order: порядок сортировки
    :return: список детекций
    """
    clauses = []
    params: list = []
    if status:
        clauses.append("status = %s")
        params.append(status)
    if camera_id is not None:
        clauses.append("camera_id = %s")
        params.append(camera_id)
    if time_from:
        clauses.append("detection_time >= %s")
        params.append(time_from)
    if time_to:
        clauses.append("detection_time <= %s")
        params.append(time_to)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"""
        SELECT detection_id, camera_id, detection_time, status
        FROM detections
        {where}
        ORDER BY detection_time {order}
        LIMIT %s
    """
    params.extend([limit])
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
            return [Detection(detection_id=r[0], camera_id=r[1], detection_time=r[2], status=r[3]) for r in rows]


@helmet_db_api.get("/detections/{detection_id}", response_model=Detection)
async def get_detection(detection_id: int):
    """
    Возвращает информацию о детекции.

    :param detection_id: ID детекции
    :return: объект детекции
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT detection_id, camera_id, detection_time, status FROM detections WHERE detection_id=%s",
                (detection_id,),
            )
            row = await cur.fetchone()
            if not row:
                raise HTTPException(404, "Not found")
            return Detection(detection_id=row[0], camera_id=row[1], detection_time=row[2], status=row[3])


@helmet_db_api.get("/detections/{detection_id}/image")
async def get_detection_image(detection_id: int):
    """
    Возвращает изображение детекции.

    :param detection_id: ID детекции
    :return: поток изображения
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT image FROM detections WHERE detection_id=%s", (detection_id,))
            row = await cur.fetchone()
            if not row:
                raise HTTPException(404, "Not found")
            data: bytes = row[0]
            return StreamingResponse(iter([data]), media_type="image/jpeg")


@helmet_db_api.patch("/detections/{detection_id}/status", response_model=Detection)
async def update_status(detection_id: int, body: DetectionStatus):
    """
    Обновляет статус детекции.

    :param detection_id: ID детекции
    :param body: новый статус
    :return: обновлённая детекция
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE detections SET status=%s WHERE detection_id=%s RETURNING detection_id, camera_id, detection_time, status",
                (body.status, detection_id),
            )
            row = await cur.fetchone()
            if not row:
                raise HTTPException(404, "Not found")
            return Detection(detection_id=row[0], camera_id=row[1], detection_time=row[2], status=row[3])


@helmet_db_api.delete("/admin/delete_outdated")
async def delete_outdated():
    """
    Удаляет устаревшие записи.

    :return: статус операции
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM detections WHERE status='rejected'")
            return {"ok": True}


if __name__ == "__main__":
    # WindowsSelectorEventLoopPolicy — требуется для корректной работы uvicorn под Windows
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run("db.helmet_db_api:helmet_db_api", reload=True)
