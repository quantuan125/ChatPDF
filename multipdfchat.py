import streamlit as st 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from htmlTemplates import css, bot_template, user_template
import base64
from base64 import b64encode
import os.path
import tempfile
from tempfile import NamedTemporaryFile
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from typing import Dict, Any
import os
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging 
import torch 
from langchain.llms import HuggingFacePipeline, LlamaCpp
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from huggingface_hub import hf_hub_download
from InstructorEmbedding import INSTRUCTOR


def display_pdfs(file_path):
            with open(file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = F'<portal src="data:application/pdf;base64,{base64_pdf}" width="600" height="700"></portal>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):

    headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(text)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_documents(md_header_splits)
    return docs

def get_vectorstore(text_chunks):
    texts = [doc.page_content for doc in text_chunks]
    device_type = "cpu"
    if 'openai' in st.session_state and st.session_state.openai is not None:
        embeddings = OpenAIEmbeddings()
    elif 'local' in st.session_state and st.session_state.local is not None:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
    return vectorstore

class AnswerConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(inputs,{'response': outputs['answer']})

def get_conversation_chain(vectorstore):
    if 'openai' in st.session_state and st.session_state.openai:
        llm = ChatOpenAI(temperature=0.5)
    elif 'local' in st.session_state and st.session_state.local:
        llm = st.session_state.local
    memory = AnswerConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        #combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        chain_type="stuff",
        memory=memory,
        return_source_documents = True
    )
    return conversation_chain

def get_qa_chain(vectorstore):
    if 'openai' in st.session_state and st.session_state.openai:
        llm = ChatOpenAI(temperature=0)
    elif 'local' in st.session_state and st.session_state.local:
        llm = st.session_state.local
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_PROMPT},
        chain_type="stuff",
        return_source_documents = True
    )
    return qa_chain

def qa_handle_userinput():
    user_question = st.session_state.selected_question if st.session_state.selected_question else st.session_state.user_question
    if user_question is not None and st.session_state.qa is not None:
        qa_response = st.session_state.qa({'query': user_question})
        st.session_state.source_documents = qa_response['source_documents']

        st.subheader("Query")
        with st.expander("Question", expanded = True):
            st.markdown(qa_response["query"])
        st.subheader("Result")
        with st.expander("Answer", expanded = True):
            st.markdown(qa_response["result"])
        with st.expander("Source"):
            formatted_source = qa_response["source_documents"][0].page_content.replace('\\n', '\n')
            st.markdown(f"**<u>Text:</u>**<br>{formatted_source}", unsafe_allow_html=True)
        st.session_state.selected_question = None

def chat_handle_userinput():
    user_question = st.session_state.selected_question if st.session_state.selected_question else st.session_state.user_input
    if user_question is not None and st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        st.session_state.source_documents = response['source_documents']
        st.session_state.selected_question = None
    
    if st.session_state.get('chat_history'):
        chat_history_reversed = reversed(st.session_state.chat_history)
        for i, message in enumerate(chat_history_reversed):
            if i % 2 == 0:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)

def clear_text():
     st.session_state.text = ""
     st.session_state.summary = None
     st.session_state.pdf = None
     st.session_state.txt = None
     st.session_state.source_documents = None
     st.session_state.user_question = None
     st.session_state.sample_questions = None

def custom_summary(docs, type):

    map_prompt_template = """ Write a summary of this chunk of text that includes the main points and any important details.
    {text}
    """

    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    combine_prompt_template = """
    Write a concise summary of the following text delimited by triple backquotes. This summary should take in the form of {type} that outline the key points of the text:
    ```{text}```
    """
    comebine_prompt = PromptTemplate(template = combine_prompt_template, input_variables=["text","type"])
    chain = load_summarize_chain(
            llm = ChatOpenAI(temperature=0), 
            chain_type="map_reduce", 
            combine_prompt=comebine_prompt,
            map_prompt=map_prompt,
            token_max = 3000
            )
    
    result_summary = chain({"input_documents": docs, "type": type}, return_only_outputs=True)["output_text"]
    return result_summary

def generate_sample_questions(text_chunks, num_questions):
    prompt_template = "Given the following text, generate a question that can be answered using the information in the text:\n\n{text}"
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    llm = ChatOpenAI(temperature=0.5)
    question_generation_chain = LLMChain(llm=llm, prompt=prompt)
    questions = []
    for chunk in text_chunks:
        question = question_generation_chain({'text': chunk.page_content})
        questions.append(question['text'])
    if len(questions) > num_questions:
        questions = questions[:num_questions]
    return questions

def load_local_model(device_type, model_id, model_path, model_basename=None):
    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for GGML quantized models")
            model_path = model_path
            max_ctx_size = 2048
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)

        else:
            if ".safetensors" in model_basename:
                # Remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")

            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            logging.info("Tokenizer loaded")

            model = AutoGPTQForCausalLM.from_quantized(
                model_path,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
    elif (
        device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    generation_config = GenerationConfig.from_pretrained(model_path)
    # Create a pipeline for text generation

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config
    )
    #st.write(tokenizer)
    #st.write(model_basename)

    local_model = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_model

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)
    
    #If a exists then ignore and continue, if a does not exists then equals and run 1 time
    if "qa" not in st.session_state: 
        st.session_state.qa = None
    if "conversation" not in st.session_state: 
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = None
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "txt" not in st.session_state: 
        st.session_state.txt = None
    if "pdf" not in st.session_state: 
        st.session_state.pdf = None
    if "user_input" not in st.session_state: 
        st.session_state.user_input = None
    if "source_documents" not in st.session_state: 
        st.session_state.source_documents = None
    if "qa_format" not in st.session_state:
        st.session_state.qa_format = None
    if "chat_format" not in st.session_state:
        st.session_state.chat_format = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = None
    if "sample_questions" not in st.session_state:
        st.session_state.sample_questions = None
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = None
    if "openai" not in st.session_state:
        st.session_state.openai = None
    if "local" not in st.session_state:
        st.session_state.local = None
    

    base_directory = os.getcwd()

    st.header("Chat with multiple PDFs :books:")
    with st.expander("Configuration", expanded = True):
        llm_selection = st.selectbox("Select LLM Model", ("OpenAI", "LocalAI"))
        if llm_selection == "OpenAI":
            openai_api_key = st.text_input("Enter OpenAI API Key", value="", placeholder="Enter the OpenAI API key which begins with sk-", type="password")
            if openai_api_key:
                st.session_state.openai = openai_api_key
                os.environ["OPENAI_API_KEY"] = openai_api_key
                st.write("API key has entered")

        elif llm_selection == "LocalAI":
            localai_model = st.selectbox("Select LocalAI Model", ("None", "Llama-2-7B-Chat-GGML", "Llama-2-7B-Chat-GPTQ"))
            if localai_model != "None":
                if localai_model == "Llama-2-7B-Chat-GGML":
                    model_path = os.path.join(base_directory, "local", "Llama2", "7B", "GGML", "llama-2-7b-chat.ggmlv3.q4_0.bin")
                    model_id = "TheBloke/Llama-2-7B-Chat-GGML"
                    model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"
                    device_type = "cpu"
                elif localai_model == "Llama-2-7B-Chat-GPTQ":
                    model_path = os.path.join(base_directory, "local", "Llama2", "7B", "GPTQ")
                    model_id = "TheBloke/Llama-2-7B-Chat-GPTQ"
                    model_basename = "gptq_model-4bit-128g.safetensors"
                    device_type = "cuda:0"
                st.session_state.local = load_local_model(device_type=device_type, model_id=model_id,  model_path=model_path, model_basename=model_basename)
                st.write("Local LLM model has been loaded. Press 'Process' to continue")
            st.write(st.session_state.local)

    pdf_docs = None

    if 'openai' in st.session_state and st.session_state.openai or 'local' in st.session_state and st.session_state.local:
        with st.sidebar:
            st.subheader("Format Type")
            format = st.radio("Choose The Format",('QA','Chat'), horizontal=True, help="Each format selection requires pressing 'Process' the first time")
            if format == 'QA':
                st.session_state.qa_format = True
                st.session_state.chat_format = False
            elif format == 'Chat':
                st.session_state.chat_format = True
                st.session_state.qa_format = False
            st.subheader("Your documents")
            pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    if pdf_docs is not None:
                        for doc in pdf_docs:
                            file_details = {"FileName":doc.name,"FileType":doc.type,"FileSize":doc.size}
                            #st.write(file_details)
                        if doc.type == "application/pdf":
                                #get pdf text
                                raw_text = get_pdf_text(pdf_docs)
                                st.session_state.pdf = raw_text

                                #get pdf text chunks
                                st.session_state.text_chunks = get_text_chunks(raw_text)
                                #st.write(st.session_state.text_chunks)

                                #create vector store 
                                vectorstore = get_vectorstore(st.session_state.text_chunks)

                                if st.session_state.qa_format is True:
                                    #create qa chain 
                                    st.session_state.qa = get_qa_chain(vectorstore)

                                if st.session_state.chat_format is True:
                                    #create conversation chain 
                                    st.session_state.conversation = get_conversation_chain(vectorstore)


                        elif doc.type == "text/plain":
                                #get txt text
                                raw_text = str(doc.read(),"utf-8")
                                st.session_state.txt = raw_text
                                #get txt text chunks
                                st.session_state.text_chunks = get_text_chunks(raw_text)

                                #create vector store 
                                vectorstore = get_vectorstore(st.session_state.text_chunks)

                                if st.session_state.qa_format is True:
                                    #create qa chain 
                                    st.session_state.qa = get_qa_chain(vectorstore)

                                if st.session_state.chat_format is True:
                                    #create conversation chain 
                                    st.session_state.conversation = get_conversation_chain(vectorstore)

            type_summary = st.selectbox('Type of Summary',('Paragraph', 'Few Bullet Points'), key="type_summary")
            if st.button("Summarize"): 
                with st.spinner("Summarizing..."):
                    #create summary
                    st.session_state.summary = custom_summary(st.session_state.text_chunks, type_summary)
            num_questions = st.slider("Number of Questions", min_value=1, max_value=10, step=1, value=3)
            if st.button("Generate Sample Questions"):
                with st.spinner("Summarizing..."):
                    st.session_state.sample_questions = generate_sample_questions(st.session_state.text_chunks, num_questions=num_questions)



        with st.expander("Summarization"):
            if st.session_state.summary is not None:
                result_summary = st.session_state.summary
                st.write(result_summary)
        with st.expander("Sample Questions"):
            if st.session_state.sample_questions is not None:
                for i, question in enumerate(st.session_state.sample_questions):
                    if st.button(question, key=f"question_{i}"):
                        st.session_state.selected_question = question


    st.session_state.user_input = st.chat_input("Ask a question about your documents:")
    if st.session_state.user_input:
        st.session_state.user_question = st.session_state.user_input

    reset_button = st.button("Reset Chat", on_click=clear_text)
    if reset_button:
            st.session_state.conversation = None
            st.session_state.chat_history = None
            st.session_state.qa = None
            st.session_state.user_question = None
            st.warning("Chat has been reset. Click on 'Process' again to start another conversation")
        
    
    col1,col2,col3 = st.columns([2,1,3], gap="small")
    
    with col1:
        if pdf_docs is not None:
                for doc in pdf_docs:
                    if doc.type == "application/pdf":
                        temp_dir = tempfile.TemporaryDirectory()
                        tmp_dir_path = temp_dir.name
                        file_path = os.path.join(tmp_dir_path, doc.name)
                        file = open(file_path, "wb")
                        file.write(doc.getbuffer())
                        if st.session_state.pdf is not None:
                            if st.session_state.qa or st.session_state.conversation is not None:
                                display_pdfs(file_path)
                                st.write(doc.name)
                            

                    elif doc.type == "text/plain":
                        st.write(st.session_state.txt)
            
        else:
                file_path = ""

        
    with col3:
        #st.write(st.session_state.user_input)
        #st.write(st.session_state.user_question)
        if st.session_state.qa_format == True:
            user_question = st.session_state.user_question
            if user_question:
                qa_handle_userinput()
            else:
                qa_handle_userinput()

        if st.session_state.chat_format == True:
            user_question = st.session_state.user_input
            if user_question:
                chat_handle_userinput()
            else:
                chat_handle_userinput()
    
    if st.session_state.chat_format is True and st.session_state.conversation is not None:
        with st.expander("Sources"):
            source_documents = st.session_state.source_documents
            if source_documents:
                for i, doc in enumerate(source_documents):
                    formatted_source = doc.page_content.replace('\\n', '\n')
                    st.markdown(f"**<u>Text {i+1}:</u>**<br>{formatted_source}", unsafe_allow_html=True)


if __name__== '__main__':
    main()
