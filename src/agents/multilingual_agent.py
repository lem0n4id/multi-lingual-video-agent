# Necessary imports: uagents for agent creation and message handling,
# os and requests for managing API calls
from uagents import Agent, Context
from uagents.setup import fund_agent_if_low
import os
import requests

# The access token and URL for the SAMSUM BART model, served by Hugging Face
HUGGING_FACE_ACCESS_TOKEN = os.getenv(
    "HUGGING_FACE_ACCESS_TOKEN", "HUGGING FACE secret phrase :)")
SAMSUM_BART_URL = "https://api-inference.huggingface.co/models/Samuela39/my-samsum-model"

# Setting the headers for the API call
HEADERS = {
    "Authorization": f"Bearer {HUGGING_FACE_ACCESS_TOKEN}"
}

SEED=HUGGING_FACE_ACCESS_TOKEN

# Creating the agent and funding it if necessary
agent = Agent(
    name="multilingual-agent",
    seed=SEED,
    port=8001,
    endpoint=["http://127.0.0.1:8001/submit"],
)
fund_agent_if_low(agent.wallet.address())

@agent.on_event("startup")
async def on_start(ctx: Context):
    ctx.logger.info("Multilingual agent started")
    data = {
        "inputs": "Amanda: I baked  cookies. Do you want some more?\r\nJerry: Sure!\r\nAmanda: I'll bring you tomorrow :-)"
    }

    try:
        # Making POST request to Hugging Face SAMSUM BART API
        response = requests.post(SAMSUM_BART_URL, headers=HEADERS, json=data)

        if response.status_code != 200:
            # Error handling - send error message back to user if API call unsuccessful
            await ctx.logger.error(f"Error: {response.json().get('error')}")
            return
        # If API call is successful, return the response from the model
        model_res = response.json()[0]
        ctx.logger.info(f"Response from model: {model_res}")
        return
    except Exception as ex:
        # Catch and notify any exception occured during API call or data handling
        await ctx.logger.error(f"An exception occurred while processing the request: {ex}")
        return

if __name__ == "__main__":
    agent.run()