from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*allowed_objects.*", category=PendingDeprecationWarning)

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langsmith import traceable
from pydantic import BaseModel, Field, ValidationError

load_dotenv(override=True)
MAX_ITERATIONS = 6
MODEL_NAME = "openai/gpt-oss-120b"
MODEL_PROVIDER = "groq"
PRICES = {"laptop": 100.5, "printer": 55.2, "headphones": 27.8}
DISCOUNT_PERC = {'bronze': 5, 'silver': 10, 'gold': 25}


# Output Structure
class AgentResponse(BaseModel):
    """Return the final answer to the user.

    Call this ONLY after every required tool has already run and you are holding
    real numbers returned by those tools. Never call this on the first turn and
    never populate it with values you calculated or guessed yourself.
    """
    product_name: str = Field(description="Name of the product")
    discounted_price: float = Field(description="Final discounted price of the product.")


## Model

llm = init_chat_model(
    model=MODEL_NAME,
    model_provider=MODEL_PROVIDER,
    temperature=0
)


## tools

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in a catalog"""
    print(f"Executing get_product_price for product: '{product}'")
    return PRICES.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to the price and return final price.
    Available tiers: bronze, silver, gold."""
    print(f"Executing apply_discount for price: '{price}' and product tier: '{discount_tier}'")
    return round(price * (1 - DISCOUNT_PERC.get(discount_tier, 0) / 100), 2)


tools = [get_product_price, apply_discount]
tools_dict = {t.name: t for t in tools}

# The response schema is bound alongside the real tools, so the model
# chooses each turn between "do more work" and "answer".
llm_with_tools = llm.bind_tools([*tools, AgentResponse])


## Agent Loop

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):

    messages = [
        SystemMessage(content="""You are a helpful shopping assistant. 
                                 You have access to a product catalog tool
                                 and a discount tool\n\n
                                 STRICT RULES: you must follow them exactly: \n
                                 1. Never guess or assume any product price. you must call the get_product_price to get the real price. \n
                                 2. Only call apply_discount after you have received the price from get_product_price execution. do not pass a made up number. \n
                                 3. Never guess the discount amount. you must call the apply_discount to get the exact discount. Never calculate discounts yourself. \n
                                 4. If the user does not specify a discount tier - ask them which tier to apply. Never assume this. \n
                                 5. When you have the final discounted price from apply_discount, call AgentResponse to return the answer."""),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n -- Iteration -- {iteration} --")

        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        tool_calls = ai_message.tool_calls

        # No tool call at all -> plain text. 
        if not tool_calls:
            print(f"Plain text reply: {ai_message.content}")
            return ai_message.content

        final_result = None

        # Every tool_call in the turn needs a matching ToolMessage, or the next
        # invoke() will be rejected by the provider.
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")
            print(f"Selected tool: {tool_name} with args: {tool_args} and ID: {tool_call_id}")

            if tool_name == "AgentResponse":
                try:
                    final_result = AgentResponse(**tool_args)
                    observation = "Final response accepted."
                except ValidationError as e:
                    observation = (
                        f"AgentResponse validation failed: {e}\n"
                        "Correct the arguments and call AgentResponse again."
                    )
                    print(f"[Validation Error]: {e}")
            else:
                tool_to_use = tools_dict.get(tool_name)
                if tool_to_use is None:
                    observation = f"Error: tool '{tool_name}' not found."
                else:
                    try:
                        observation = tool_to_use.invoke(tool_args)
                    except Exception as e:
                        observation = f"Error running {tool_name}: {e}"
                print(f"[Tool Result]: {observation}")

            messages.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )

        if final_result is not None:
            # print(f"[Structured Response]: {final_result}")
            return final_result

    print("Error: MAX Iterations crossed without a final answer")
    return None


if __name__ == "__main__":
    response = run_agent("What is the price of a laptop after applying a gold discount?")
    print("\n", response)

    if isinstance(response, AgentResponse):
        print(response.product_name, response.discounted_price)