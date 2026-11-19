import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv(dotenv_path=".env")
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1beta"},
)
MODEL = "gemini-2.5-flash"

def echo(text: str) -> dict:
    return {"echoed": text.upper()}

echo_decl = types.FunctionDeclaration(
    name="echo",
    description="Echo text in uppercase.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={"text": types.Schema(type=types.Type.STRING)},
        required=["text"],
    ),
)

contents = [types.Content(
    role="user",
    parts=[types.Part.from_text(text="Call the echo tool with 'hello world'.")],
)]

for i in range(4):
    r = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=[echo_decl])],
            temperature=0,
        ),
    )
    if not r.function_calls:
        print("phase1 done, text:", r.text)
        break
    contents.append(r.candidates[0].content)
    for fc in r.function_calls:
        print("tool:", fc.name, dict(fc.args))
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_function_response(
                name=fc.name,
                response=echo(**dict(fc.args)),
            )],
        ))

class Answer(BaseModel):
    result: str
    explanation: str

contents.append(types.Content(
    role="user",
    parts=[types.Part.from_text(text="Summarize as Answer.")],
))

r2 = client.models.generate_content(
    model=MODEL,
    contents=contents,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Answer,
        temperature=0,
    ),
)
print("phase2 text:", r2.text)
print("phase2 parsed:", r2.parsed)
