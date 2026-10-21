from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

class CelestialBody(BaseModel):
  name: str = Field(description="天体の名前（漢字表記）")
  radius: float = Field(description="天体の半径（ km ）")
  mass: float = Field(description="天体の質量（ kg ）")
  type: Literal["恒星", "惑星", "衛星"] = Field(description="天体の種類")

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
llm_with_structured_output = llm.with_structured_output(CelestialBody)
prompt_template = PromptTemplate.from_template("{planet}の情報を教えてください。")

venus = llm_with_structured_output.invoke(prompt_template.invoke({"planet": "金星"}))
earth = llm_with_structured_output.invoke(prompt_template.invoke({"planet": "地球"}))

print(venus)
print(earth)