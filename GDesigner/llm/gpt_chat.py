import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os

from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry

load_dotenv()

LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai_compatible').strip().lower()
BASE_URL = os.getenv('BASE_URL', '').strip()
API_KEY = os.getenv('API_KEY', '').strip()


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict],):
    request_url = BASE_URL
    messages = [{"role": m.role, "content": m.content} if isinstance(m, Message) else m for m in msg]

    if LLM_PROVIDER == 'ollama':
        headers = {'Content-Type': 'application/json'}
        data = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(request_url, headers=headers, json=data) as response:
                response.raise_for_status()
                response_data = await response.json()
                answer = response_data.get('message', {}).get('content', '')
                prompt = "".join([item['content'] for item in messages])
                if answer:
                    cost_count(prompt, answer, model)
                return answer

    headers = {'Content-Type': 'application/json'}
    if API_KEY:
        headers['authorization'] = API_KEY

    data = {
        "name": model,
        "inputs": {
            "stream": False,
            "msg": repr(messages),
        }
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(request_url, headers=headers, json=data) as response:
            response.raise_for_status()
            response_data = await response.json()
            prompt = "".join([item['content'] for item in messages])
            cost_count(prompt, response_data['data'], model)
            return response_data['data']


@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name, messages)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass
