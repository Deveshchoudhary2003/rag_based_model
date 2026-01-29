from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    video_id: str = Field(
        ...,
        min_length=5,
        description="YouTube video ID",
        example="yKeNBjo_lJU"
    )
    question: str = Field(
        ...,
        min_length=3,
        description="User question based on video transcript",
        example="What is FAISS?"
    )
