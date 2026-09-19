from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI(title="Math API")


@app.get("/add")
def add(a: float, b: float) -> dict:
    return {"result": a + b}


@app.get("/subtract")
def subtract(a: float, b: float) -> dict:
    return {"result": a - b}


@app.get("/multiply")
def multiply(a: float, b: float) -> dict:
    return {"result": a * b}


@app.get("/divide")
def divide(a: float, b: float) -> dict:
    if b == 0:
        raise HTTPException(status_code=400, detail="Cannot divide by zero")
    return {"result": a / b}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
