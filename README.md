# Schemallm

An intuitive and robust way to generate structured text. Inspired by [Jsonformer](https://github.com/1rgs/jsonformer)

## Base functions

supported schema types:
- number
- boolean
- string
- array
- object

## What's new

- Pydantic is supported
- Literal is supported
- Tuple is supported

# Usage

```python
# define schema
from typing import Tuple
from pydantic import BaseModel, Field

class Employer(BaseModel):
    name: str
    location: str

class Person(BaseModel):
    name: str
    age: int
    employer: Employer
    job: str
    family_members: Tuple[str, str] 
    # or
    # family_members: List[str] = Field(minItems=2, maxItems=2)

# load model
from schemallm import SchemaLLM
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GPTQ")
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-Chat-GPTQ")
schema_llm = SchemaLLM(
    model=model,
    tokenizer=tokenizer
)
output = schema_llm.generate(prompt=input_text, schema=schema)
print(output)
# {'name': 'Alice', 'age': 21, 'employer': {'name': 'LMQL Inc', 'location': 'Zurich'}, 'job': 'engineer', 'family_members': ['husband', 'son']}
```

# Install

`pip install git+https://github.com/pfan21/schemallm`
