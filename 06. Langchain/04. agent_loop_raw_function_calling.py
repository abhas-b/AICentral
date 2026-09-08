from groq import Groq
import json
from dotenv import  load_dotenv

load_dotenv(override=True)
client = Groq()
MAX_ITERATIONS = 2
MODEL_NAME = "openai/gpt-oss-120b"
PRICES = {"laptop": 100.5, "printer": 55.2, "headphones": 27.8}
DISCOUNT_PERC = {'bronze': 5, 'silver': 10, 'gold': 25}


def get_product_price(product: str) -> str:
    """Look up the price of a product in a catalog"""
    print(f"Executing get_product_price for product: '{product}'")
    return json.dumps({"price": PRICES.get(product, 0)})


def apply_discount(discount_tier: str, price: float) -> float:
    """Apply a discount tier to the price and return final price.
    Available tiers: bronze, silver, gold."""
    print(f"Executing apply_discount for price: '{price}' and product tier: '{discount_tier}'")
    return json.dumps({"discounted_price": round(price * (1 - DISCOUNT_PERC.get(discount_tier, 0) / 100), 2)})


available_functions = {"get_product_price": get_product_price,
                       "apply_discount": apply_discount}


tools = [
        {
            "type": "function",
            "function": {
                "name": "get_product_price",
                "description": """Look up the price of a product in a catalog""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product": {
                            "type": "string",
                            "description": "The product name as mentioned in the catalog",
                        }
                    },
                    "required": ["product"],
                },
            },
        },

        {
                    "type": "function",
                    "function": {
                        "name": "apply_discount",
                        "description": """Apply a discount tier to the price and return final price.
                            Available tiers: bronze, silver, gold.""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "discount_tier": {
                                    "type": "string",
                                    "description": "The discount tier: gold, silver, bronze",
                                },
                                "price": {
                                    "type": "number",
                                    "description": "Price before discount",
                                                                }
                            },
                            "required": ["discount_tier", "price"],
                        },
                    },
                },

    ]



def run_agent(question: str):
    """Run an agent with tool calling"""
    messages = [{"role":"system", "content": """You are a helpful shopping assistant. 
                                    You have access to a product catalog tool
                                    and a discount tool\n\n
                                    STRICT RULES: you must follow them exactly: \n
                                    1. Never guess or assume any product price. you must call the get_product_price to get the real price. \n
                                    2. Only call apply_discount after you have received the price from get_product_price execution. do not pass a made up number. \n
                                    3. Never guess the discount amount. you must call the apply_discount to get the exact discount. Never calculate discounts yourself. \n
                                    4. If the user does not specify a discount tier - ask them which tier to apply. Never assume this. \n
                                    5. When you have the final discounted price from apply_discount, call AgentResponse to return the answer."""},
            {"role": "user", "content": question}
        ]

    # Initial request
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"--- Iteration: {iteration} ---")

        response_message = response.choices[0].message
        print(messages)
        messages.append(response_message)

        if not response_message.tool_calls:
            break

        for tool_call in response_message.tool_calls:
            tool_name = tool_call.function.name
            tool_to_call = available_functions[tool_name]
            tool_args = json.loads(tool_call.function.arguments)
            tool_response = tool_to_call(**tool_args)
            print(tool_name)
            messages.append(
                {
                    "role":"tool",
                    "tool_call_id":tool_call.id,
                    "name":tool_name,
                    "content":tool_response
                }
            )
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        print()

    print(f"Assistant: {response.choices[0].message}")




if __name__=="__main__":
    run_agent("What is the price of a laptop after applying a gold discount?")
