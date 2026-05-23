from utils.llm_engine import (
    generate_llm_response
)

response = generate_llm_response(
    "The user spends too much money when stressed."
)

print(response)