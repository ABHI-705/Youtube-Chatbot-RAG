from langdetect import detect
from langchain_google_genai import ChatGoogleGenerativeAI

llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

def extract_id(url:str):
    video_id=url.split("v=")[1]
    return video_id

def language_detector(text):
    detected_lang=detect(text)
    return detected_lang

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

def translator(text):
    detected_lang=detect(text)

    if detected_lang=="en":
        return text, detected_lang
    else:
        prompt = f"""You are a professional translator. 
    Translate the following {detected_lang} text into fluent and natural English.
    Preserve the original meaning as accurately as possible.

    Text:
    {text}
    """
    response=llm.invoke(prompt)
    translated_text=response.content

    return translated_text, detected_lang