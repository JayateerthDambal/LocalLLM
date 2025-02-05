from fastapi import FastAPI
import time
from app.core.model_handler import handler

model_app = FastAPI(title="LLM Model Chat App (DeepSeekModel)")

# TODO: Later we will add the user selected model to initialized it.


@model_app.get("/initialize-model", summary="Initalizes the selected model")
async def initialize_model():
    """
    USECASE: To initialized and warm-up the selected model
    if the model is initialized, it returns a message
    """
    start_time = time.time()

    result = handler.initialize_model()
    elapsed_time = time.time() - start_time
    result["time_taken"] = elapsed_time

    return result


@model_app.get("/deinitialize", summary="Unload the Model and Clear GPU Memory")
async def deinitialize_model():
    result = handler.deinitialize_running_model()
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:model_app",
                host="0.0.0.0", port=8000, reload=True)
