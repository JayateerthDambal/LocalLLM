from fastapi import FastAPI, HTTPException
import time
from core.model_handler import handler
from pydantic import BaseModel
from core.session_manager import session_manager


app = FastAPI(title="LLM Model Chat App (DeepSeekModel)")

# TODO: Later we will add the user selected model to initialized it.


@app.get("/initialize-model", summary="Initalizes the selected model")
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


@app.get("/deinitialize", summary="Unload the Model and Clear GPU Memory")
async def deinitialize_model():
    result = handler.deinitialize_running_model()
    return result


# * -------- CHAT Sessions Endpoints -------- * #
class StartSessionResponse(BaseModel):
    session_id: str
    message: str


@app.post("/start-session", response_model=StartSessionResponse, summary="Start a new chat session")
async def start_session():
    """
    Creates a new chat session and returns an unique id
    """

    session_id = session_manager.create_session()

    return {"session_id": session_id, "message": "Session started"}


class Chatrequest(BaseModel):
    session_id: str
    message: str


class ChatReponse(BaseModel):
    response: str
    conversation_list: list


@app.post("/chat", response_model=ChatReponse, summary="Send a message to a chat session")
async def chat(request: Chatrequest):
    """
    Apppends a user message to the session history, uses this list as a
    context, generate a response from the model, and update the session with 
    model's reply.
    """
    # Gert the session_id
    session = session_manager.get_session(request.session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    session_manager.add_message(request.session_id, "user", request.message)

    messages = session_manager.get_session(request.session_id)

    model = handler.get_model()
    tokenizer = handler.get_tokenizer()
    start_time = time.time()
    # Format the converstaion histary as input for the model
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors='pt'
    ).to(model.device)

    # Generate the response:
    outputs = model.generate(
        inputs,
        max_length=1024,
        do_sample=True,
        top_k=50,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    response_text = tokenizer.decode(
        outputs[0][len(inputs[0]):], skip_special_tokens=True
    )

    session_manager.add_message(request.session_id, "assistant", response_text)
    elapsed_time = time.time() - start_time
    return {
        "time_taken": elapsed_time,
        "response": response_text,
        "conversation_list": session_manager.get_session(request.session_id)
    }


# @app.on_event("shutdown")
# async def shutdown_event():
#     handler.deinitialize_running_model()
#     print("Model deinitialized durng shutdown!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app",
                host="0.0.0.0", port=8000, reload=True)
