import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- Step 1: Define a real Python function ---
def calculate(expression: str) -> str:
    """Evaluates a math expression and returns the result."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# --- Step 2: Tell the model this tool exists ---
calculator_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="calculate",
            description="Evaluates a mathematical expression. Use this for any math calculation.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "expression": types.Schema(
                        type=types.Type.STRING,
                        description="The math expression to evaluate e.g. '25 * 4 + 10'"
                    )
                },
                required=["expression"]
            )
        )
    ]
)

# --- Step 3: Send a message and let the agent decide ---
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is 15% tip on a bill of 3500 rupees?",
    config=types.GenerateContentConfig(
        tools=[calculator_tool],
        system_instruction="You are a helpful assistant. Answer general questions directly from your knowledge. Only use the calculate tool for math problems."
    )
)    

# --- Step 4: Check if the model wants to use the tool ---
part = response.candidates[0].content.parts[0]

if part.function_call:
    tool_name = part.function_call.name
    tool_args = part.function_call.args
    print(f"Agent decided to use tool: {tool_name}")
    print(f"With arguments: {tool_args}")

    # --- Step 5: We run the actual function ---
    result = calculate(tool_args["expression"])
    print(f"Tool returned: {result}")

    # --- Step 6: Send result back to model for final answer ---
    final_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part(text="What is 15% tip on a bill of 3500 rupees?")]),
            types.Content(role="model", parts=[types.Part(function_call=part.function_call)]),
            types.Content(role="user", parts=[types.Part(
                function_response=types.FunctionResponse(
                    name=tool_name,
                    response={"result": result}
                )
            )])
        ],
        config=types.GenerateContentConfig(tools=[calculator_tool])
    )
    print(f"\nFinal answer: {final_response.text}")
else:
    print(f"Agent answered directly: {part.text}")