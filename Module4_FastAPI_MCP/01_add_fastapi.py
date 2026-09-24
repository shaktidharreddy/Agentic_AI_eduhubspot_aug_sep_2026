from fastapi import FastAPI

app = FastAPI()


@app.get("/add")
def add(a: int, b: int):
    return {"result": a + b}

@app.get("/multiply")
def multiply(a: int, b: int):
    """Multiply two numbers."""
    return {"result": a * b}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)