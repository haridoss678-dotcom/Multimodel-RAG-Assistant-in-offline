from fastapi import APIRouter, HTTPException

from rag_assistant.storage.db import get_connection

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}")
def get_job(job_id: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT job_id, stage, progress_pct, error FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(
            status_code=404, detail={"code": "NOT_FOUND", "message": "job not found"}
        )
    return dict(row)
