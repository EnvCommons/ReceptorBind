"""Test agent for the ReceptorBind environment.

Runs a single task against a local server using the OpenAI Responses API.
"""

import json
import asyncio
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

MODEL_NAME = "gpt-5.2"
ENV_NAME = "GeneralReasoning/ReceptorBind"
SPLIT = "test"
BASE_URL = "http://localhost:8080"


async def main():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")

    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=openai_api_key)

    environment = or_client.environments.get(name=ENV_NAME, base_url=BASE_URL)
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks in {SPLIT} split")

    for task in tasks[:1]:
        print(f"\n--- Task: {task.task_spec.get('task_id', 'unknown')} ---")

        async with environment.session(task=task, secrets={}) as session:
            prompt = await session.get_prompt()
            prompt_text = prompt[0].text if prompt else ""

            print(f"Prompt preview: {prompt_text[:200]}...")
            input_list = [{"role": "user", "content": prompt_text}]
            finished = False

            while not finished:
                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=input_list,
                )

                input_list += response.output

                for item in response.output:
                    if item.type == "function_call":
                        tool_result = await session.call_tool(
                            item.name, json.loads(str(item.arguments))
                        )

                        reward = tool_result.reward
                        finished = tool_result.finished

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text,
                        })

                        print(f"Tool: {item.name}")
                        print(f"Result: {tool_result.blocks[0].text}")
                        print(f"Reward: {reward:.3f}")

                        if finished:
                            print("FINISHED!")
                            break

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
