from langchain.llms.base import LLM
from typing import Iterator, Optional, List, Any, Dict
from langchain.callbacks.manager import CallbackManagerForLLMRun
import chatglm_cpp
from langchain.tools.base import BaseTool
from typing import Union
import torch
from langchain.schema import AIMessage
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Optional
from langchain.chat_models.base import ChatGeneration, BaseChatModel, ChatGenerationChunk
from transformers import (
  PreTrainedTokenizer,
  PreTrainedTokenizerFast
)
from langchain.schema import ChatResult
from langchain.schema.messages import BaseMessage

class ChatGLM3(BaseChatModel):
  model: object = None

  dtype: str = None

  device: str = None

  max_length: int = 8192

  num_beams: int = 1

  do_sample: bool = True

  top_p: float = 0.8

  temperature: float = 0.8

  tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None

  def _llm_type(self) -> str:
    return "ChatGLM3"
  
  @classmethod
  def format_tools(history: List[BaseMessage], tools: List[BaseTool]):
    pass

  @classmethod
  def from_model_id(cls, model_id: str, dtype: str = None, **kwargs: Any) -> LLM:
    if dtype:
      try:
        import chatglm_cpp
      except :
        raise ImportError("Cannot find module chatglm_cpp. Try install it with 'pip install chatglm-cpp'")
      model = chatglm_cpp.Pipeline(model_id, dtype=dtype)
      return cls(model=model, dtype=dtype)
      
    model_config = AutoConfig.from_pretrained(
      model_id,
      trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
      model_id,
      trust_remote_code=True
    )
    
    model = AutoModel.from_pretrained(
      model_id, config=model_config, trust_remote_code=True, device_map='mps'
    )

    model = model.to(
      'cuda' if torch.cuda.is_available() else
      'mps' if torch.backends.mps.is_available() else
      'cpu'
    )
    return cls(
      model=model,
      tokenizer=tokenizer
    )
  
  def _find_last_human_message(self, messages):
    message  = messages[-1]
    if message.type != 'human':
      raise 'No human message found'
    return message, messages[0:-2]

  def _generate(
    self,
    messages: List[BaseMessage],
    stop: Optional[List[str]] = ["<|user|>"],
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
  ) -> ChatResult:
    if not self.dtype:
      message, history = self._find_last_human_message(messages)
      formated_history = self._rebuild_messages(messages=history) if history else None
      response, history = self.model.chat(
          self.tokenizer, query=message.content, history=formated_history)
      generations = ChatGeneration(
        message=AIMessage(content=response)
      )
    else:
      response = self.model.chat([
        chatglm_cpp.ChatMessage(role = message['role'], content=message['content']) 
        for message in self._rebuild_messages(messages=messages)
      ])
      generations = ChatGeneration(
        message=AIMessage(content=response.content)
      )
    return ChatResult(generations=[generations])

  
  def _stream(self, 
    messages: List[BaseMessage],
    stop: List[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None, 
    **kwargs: Any
  ) -> Iterator[ChatGenerationChunk]:
    message, history = self._find_last_human_message(messages)
    formated_history = self._rebuild_messages(
        messages=history) if history else None

    for chunk in self.mode.stream_chat(self.tokenizer, query=message.content, history = formated_history):
      yield chunk

  def _extract_role(self, message: BaseMessage):
    match message.type:
      case 'human':
        return 'user'
      case 'ai':
        return 'assistant'
      case _:
        return message.type

  def _rebuild_messages(self, messages: List[BaseMessage]) -> List[Dict]:
    return [{"role": self._extract_role(message), "content": message.content} for message in messages]
