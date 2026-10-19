from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List

class CelestialBody(BaseModel):
  name: str = Field(description="天体の名前（漢字表記）")
  radius: float = Field(description="天体の半径（ km ）")
  mass: float = Field(description="天体の質量（ kg ）")
  type: str = Field(description="天体の種類（惑星、恒星、小惑星など）")

class PlanetarySystem(BaseModel):
  planets: List[CelestialBody] = Field(description="惑星のリスト")
  center_body: CelestialBody = Field(description="中心となる恒星")
  age: float = Field(description="惑星系の年齢（億年単位）")
  name: str = Field(description="惑星系の名前")


llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
llm_with_structured_output = llm.with_structured_output(PlanetarySystem)
solar_system = llm_with_structured_output.invoke("太陽系の情報を教えてください。")

print(f"システム名：{solar_system.name}")
print(f"中心天体：{solar_system.center_body.name}")
for planet in solar_system.planets:
  print(f"・{planet.name}")
  print(f"　- 種類：{planet.radius} km")
  print(f"　- 半径：{planet.mass} kg")
  print(f"　- 質量：{planet.type}")