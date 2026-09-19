# RunnableLambda just wraps a normal function so it gains .invoke(), .batch(), .stream(), and the | pipe operator

from langchain_core.runnables import RunnableLambda

def double_my_num(x: int) -> int:
    return x * 2

# double = lambda x: x * 2
# print(double(5))   # 10

double = RunnableLambda(double_my_num)

print(double.invoke(5))   # 10

# Without the wrapper it's just a function. With it, it becomes a "runnable" that fits into a chain.


# Example 2 — piping two lambdas together

from langchain_core.runnables import RunnableLambda

add_ten   = RunnableLambda(lambda x: x + 10)
to_string = RunnableLambda(lambda x: f"Result is {x}")

chain = add_ten | to_string

print(chain.invoke(5))   # "Result is 15"

# Output of add_ten (15) flows as the input to to_string. This is the "glue between steps" role.


# Example 3 — reshaping a dict (the pattern from your file)

from langchain_core.runnables import RunnableLambda

get_name = RunnableLambda(lambda x: x["name"].upper())

print(get_name.invoke({"name": "alice", "age": 30}))   # "ALICE"


# Example 4 — using a named function instead of a lambda
# You don't have to use a lambda; any callable works. This is handy when the logic is more than one line:

from langchain_core.runnables import RunnableLambda

def clean_text(text: str) -> str:
    text = text.strip()
    text = text.replace("  ", " ")
    return text.title()

cleaner = RunnableLambda(clean_text)

print(cleaner.invoke("   hello   world  "))   # "Hello World"

# Example 5 — batch and the free benefits
# Because it's a runnable, you get .batch() for multiple inputs automatically:
from langchain_core.runnables import RunnableLambda

square = RunnableLambda(lambda x: x ** 2)

print(square.batch([1, 2, 3, 4]))   # [1, 4, 9, 16]

# Example 6 — stream and the free benefits
# Because it's a runnable, you get .stream() for streaming outputs automatically:
from langchain_core.runnables import RunnableLambda

square = RunnableLambda(lambda x: x ** 2)

for chunk in square.stream(1): print(chunk)  # 1
for chunk in square.stream(2): print(chunk) # 4
for chunk in square.stream(3): print(chunk) # 9
for chunk in square.stream(4): print(chunk)  # 16

# Example 7 — pipe and the free benefits
# Because it's a runnable, you get the | pipe operator for piping inputs automatically:
from langchain_core.runnables import RunnableLambda

square = RunnableLambda(lambda x: x ** 2)

print(square | 3)   # 9
print(square | 4)   # 16

# Example 8 — invoke and the free benefits
# Because it's a runnable, you get .invoke() for invoking the function automatically:
from langchain_core.runnables import RunnableLambda

square = RunnableLambda(lambda x: x ** 2)

print(square.invoke(5))   # 25

# Example 9 — batch and the free benefits
# Because it's a runnable, you get .batch() for multiple inputs automatically:
from langchain_core.runnables import RunnableLambda

square = RunnableLambda(lambda x: x ** 2)

print(square.batch([1, 2, 3, 4]))   # [1, 4, 9, 16]

# Example 10 — stream and the free benefits
# Because it's a runnable, you get .stream() for streaming outputs automatically:
from langchain_core.runnables import RunnableLambda

square = RunnableLambda(lambda x: x ** 2)

print(square.stream(1))   # 1
print(square.stream(2))   # 4
print(square.stream(3))   # 9
print(square.stream(4))   # 16

# Example 11 — pipe and the free benefits
# Because it's a runnable, you get the | pipe operator for piping inputs automatically: 
from langchain_core.runnables import RunnableLambda

square = RunnableLambda(lambda x: x ** 2)

print(square | 3)   # 9
print(square | 4)   # 16

# Key takeaway
# RunnableLambda = "take an ordinary Python function and let it live inside a LangChain pipeline." 
# Its whole job is single-input → single-output transformation, so it's most often used to reshape or clean data between the real steps 
# (prompt → model → parser)