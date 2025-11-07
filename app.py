from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from helper_functions import extract_id, language_detector, format_docs, translator
from langdetect import detect
from dotenv import load_dotenv
import streamlit as st
import torch


load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"
# st.success(f"Device: {device}")

embedding=HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs={"device": device})

# hf_llm=HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")
# hf_llm=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.2-3B-Instruct", task="text-generation")


# llm=ChatHuggingFace(llm=hf_llm)
llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

st.title("🎬 Youtube Chatbot")
# st.header("Youtube Chatbot")

video_id_input=st.text_input("Please paste your youtube video link here!!")

if st.button("Extract ID"):
    video_id=extract_id(video_id_input)
    if video_id:
        st.success(f"Successfully extracted the video ID: {video_id}")
        try:
            # If you don’t care which language, this returns the “best” one
            transcript_list = YouTubeTranscriptApi().fetch(video_id=video_id, languages=["en","hi"])
            print(transcript_list)

            # Flatten it to plain text
            # transcript = " ".join(chunk["text"] for chunk in transcript_list)
            transcript = " ".join(snippet.text for snippet in transcript_list)
            st.success(f"Fetched transcript successfully")
            st.text_area(f"**Transcript:**", value=transcript, height=300)
            detected_lang=detect(transcript)
            st.write(f"Detected Language: {detected_lang}")
            text=translator(transcript)
            st.text_area(f"Transcript: ", value=text[0], height=300)
            st.write("Language: English ")

            ## Indexing 
            splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
            chunks=splitter.create_documents([text[0]])
            st.write("Length of Chunks created:", len(chunks))
            st.write("Sample of chunks:", chunks[5].page_content)

            ## Embedding generation and storing in the vector store
            vector_store=FAISS.from_documents(chunks,embedding)
            st.success("Embedding Generated and stored successfully!!")
            # st.write(vector_store.index_to_docstore_id)
            # st.write(vector_store.get_by_ids(["5770a711-a992-458b-89c3-1e3c532c6de3"]))

            st.session_state["retriever"]= vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})
            
        except TranscriptsDisabled:
            print("No captions available for this video.")
        except Exception as e:
            st.error(f"Error fetching transcript: {e}")
    else:
        st.error("Could not extract video ID. Please check your video id")

# else:
#     st.warning("Please paste the Youtube Link")

if "retriever" in st.session_state:
    prompt = PromptTemplate(
    template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables = ['context', 'question']
    )
    question = st.chat_input("Ask your question:")
    if question:
        retriever = st.session_state["retriever"]
        parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
        })

        parser=StrOutputParser()

        main_chain= parallel_chain | prompt | llm | parser
        response=main_chain.invoke(question)
        st.write("Your response is here: {response}")
        st.chat_message("user").markdown(question)
        st.chat_message("assistant").markdown(response)