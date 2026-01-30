from fastapi import FastAPI, HTTPException
from main import ask_question
from input_format import QuestionRequest
from answer_format import AnswerResponse
from error_format import ErrorResponse

app = FastAPI(
    title="YouTube RAG API",
    description="Ask questions from YouTube video transcripts",
    version="1.0.0"
)

@app.get("/")
def video_check():
    return {"status": "API is running 🚀"}

@app.post(
    "/ask",
    response_model=AnswerResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
def ask_video_question(request: QuestionRequest):
    try:
        answer = ask_question(
            video_id=request.video_id,
            question=request.question
        )
        return {"answer": answer}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
