from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import sys
import base64
import os

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode("utf-8")

def explain_image(image_path: str) -> str:
  base64_image = encode_image(image_path)

  # ファイルの拡張子を取得
  ext = os.path.splitext(image_path)[1].lower()

  # 拡張子に応じたMIMEタイプの判定
  if ext == ".png":
      mime_type = "image/png"
  elif ext in [".jpg", ".jpeg"]:
      mime_type = "image/jpeg"
  elif ext == ".webp":
      mime_type = "image/webp"
  else:
      mime_type = "image/jpeg"  # デフォルト

  model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

  message = HumanMessage(
      content=[
          {"type": "text", "text": "Describe the image at the URL."},
          {
              "type": "image_url",
              "image_url": {"url": f"data:image/{mime_type};base64, {base64_image}"},
          },
      ]
  )
  response = model.invoke([message])
  return response.text

if __name__ == "__main__":
  answer = explain_image(sys.argv[1])
  print(answer)