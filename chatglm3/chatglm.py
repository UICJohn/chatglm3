from langchain.llms.base import LLM
from langchain.schema.runnable import ConfigurableField
from typing import Optional, List, Any, Dict
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.utils import enforce_stop_tokens
import torch
import os
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Optional

class ChatGLM3(LLM):
  tokenizer: object = None

  model: object = None

  dtype: str = None

  @property
  def _llm_type(self) -> str:
    return "ChatGLM3"

  @property
  def _identifying_params(self) -> Dict[str, Any]:
    """Get the identifying parameters."""
    return {**{"model": self.model, "dtype": self.dtype}}

  @classmethod
  def from_model_id(cls, model_id: str, dtype: str = None) -> LLM:
    if dtype:
      try:
        import chatglm_cpp
      except ImportError:
        raise ImportError(
          "chatglm-cpp is not installed. Please install try 'pip install chatglm-cpp'")
      
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
      model_id, config=model_config, trust_remote_code=True
    ).to(
      'cuda' if torch.cuda.is_available() else
      'mps' if torch.backends.mps.is_available() else
      'cpu'
    )
    return cls(
      model=model,
      tokenizer=tokenizer
    )

  def _call(
      self,
      prompt: str,
      stop: Optional[List[str]] = None,
      run_manager: Optional[CallbackManagerForLLMRun] = None,
      **kwargs: Any,
  ) -> str:

    if self.dtype:
      response = self.model.chat([prompt])
    else:
      response, _ = self.model.chat(
        self.tokenizer,
        prompt,
      )
    if stop:
      response = enforce_stop_tokens(response, stop)
    return response


model_id = os.getenv("CHATGLM_MODEL_ID", "./model/chatglm3-6b-32k")

dtype = os.getenv("CHATGLM_DTYPE", "q8_0")

chatglm = ChatGLM3.from_model_id(model_id=model_id, dtype=dtype)