import os
import gradio as gr
import pdfplumber

from openai import OpenAI
from groq import Groq
from supabase import create_client

# ================= CONFIG =================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


missing = []
if not OPENAI_API_KEY: missing.append("OPENAI_API_KEY")
if not GROQ_API_KEY: missing.append("GROQ_API_KEY")
if not SUPABASE_URL: missing.append("SUPABASE_URL")
if not SUPABASE_KEY: missing.append("SUPABASE_KEY")

if missing:
    raise RuntimeError(f"❌ Missing env vars: {', '.join(missing)}")


openai_client = OpenAI(api_key=OPENAI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

PDF_TEXT = ""
CHUNKS = []

# ================= PDF =================
def extract_pdf_text(file):
    text = ""
    try:
        with pdfplumber.open(file.name) as pdf:
             for page in pdf.pages:
              t = page.extract_text()
              if t:
                text += t + "\n"
    except Exception as e:
        print("PDF extraction warning:", e)
    return text


# ================= CHUNKING =================
def chunk_text(text, size=1000, overlap=200):
    chunks = []
    start = 0
    i = 0
    while start < len(text):
        end = start + size
        chunks.append({
            "id": f"chunk_{i}",
            "content": text[start:end]
        })
        start = end - overlap
        i += 1
    return chunks

# ================= EMBEDDINGS =================
def embed(text):
    res = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding

# ================= STORE =================
def store_chunks(chunks):
    for c in chunks:
        supabase.table("documents").upsert({
            "id": c["id"],
            "content": c["content"],
            "embedding": embed(c["content"]),
            "metadata": {"source": "pdf"}
        }).execute()

# ================= SEARCH =================
def search_supabase(query, k=5):
    emb = embed(query)
    res = supabase.rpc("match_documents", {
        "query_embedding": emb,
        "match_count": k
    }).execute()
    return [r["content"] for r in res.data] if res.data else []

# ================= PROMPT =================
def handbook_prompt(topic, context, section):
    return f"""
You are writing a professional technical handbook.

Topic: {topic}

Context:
{context}

Write SECTION {section}.
- Clear headings
- Detailed explanation
- No repetition
"""

# ================= STREAMING (yield HERE) =================
def generate_handbook_stream(topic, target_words=3000):
    if not topic.strip():
        yield "Please provide a meaningful topic after 'create a handbook'"
        return
    
    section = 1
    document_parts = []
    total_words = 0

    while total_words < target_words:
        context = "\n\n".join(search_supabase(topic))
        prompt = handbook_prompt(topic, context, section)

        stream = openai_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a technical writer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048,
            stream=True
        )

        section_text = ""
        for chunk in stream:
            if content:=chunk.choices[0].delta.content:
                section_text += content
                current_text = "\n\n".join(document_parts + [section_text])
                # 🔴 STREAM TOKEN-BY-TOKEN
                yield current_text
            if section_text.strip():
                document_parts.append(section_text.strip())
                total_words += len(section_text.split())
                
            section += 1
            
            if section > 25:  # Limit to 10 sections for safety
                break

        yield "\n\n".join(document_parts) + "\n\n**Handbook generation finished.**"

# ================= PDF UPLOAD =================
def upload_pdf(file):
    global PDF_TEXT, CHUNKS
    try:
        if file is None:
            return "❌ No file", ""

        PDF_TEXT = extract_pdf_text(file)
        if not PDF_TEXT.strip():
            return "⚠️ No extractable text", ""

        CHUNKS = chunk_text(PDF_TEXT)
        store_chunks(CHUNKS)

        return (
            f"✅ PDF processed\nChunks: {len(CHUNKS)}",
            CHUNKS[0]["content"][:1200]
        )
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

# ================= CHAT =================
def chat(message, history):
    if  not message.strip():
        return history
    
    current_history = history + [{"role": "user", "content": message}]
   
    if message.lower().startswith("create a handbook"):
       topic = message[15:].strip()
       if not topic:
          current_history.append({"role":"assistant", "content": "Please provide a topic for the handbook."})
          return current_history  
            
            
       thinking_msg = {"role": "assistant", "content": f"📖 Generating handbook on **{topic}** ...\n\n"}
       current_history.append(thinking_msg)
       yield current_history
        
       for partial in generate_handbook_stream(topic):
           current_history[-1]["content"] = f"📖 Handbook: **{topic}**\n\n{partial}"
           yield current_history

       return

    results = search_supabase(message,k=5)
    if not results:
       reply = "I couldn't find relevant information in the document."
    else:
       reply = "\n\n".join(results) 
       current_history.append({"role": "assistant", "content": reply})
       yield current_history

    
# ================= UI =================
with gr.Blocks(title="AI Handbook Generator") as demo:
    gr.Markdown("# 📖 AI Handbook Generator")
    gr.Markdown("PDF → Supabase RAG → Groq Streaming")
    
    with gr.Row():
        pdf = gr.File(label="Upload PDF", file_types=[".pdf"])
        status = gr.Textbox(label="Status")
    preview = gr.Textbox(label="Chunk Preview", lines=8)
    
    pdf.change(upload_pdf, pdf, [status, preview])
    
    gr.Markdown("---")
    
    chatbot = gr.Chatbot(
        height=500,
        )
    
    with gr.Row():
        msg = gr.Textbox(placeholder="Ask or: Create a handbook on ...",scale=9,lines=2)
        
        send_btn = gr.Button("Send",variant="primary",scale=1)
        
    gr.Markdown("*Hint: Try →*  `create a handbook on secure software development practices`",
        elem_classes="text-sm text-gray-500")

    msg.submit(chat, [msg, chatbot], chatbot)
    send_btn.click(chat, [msg, chatbot], chatbot)
    gr.Button("Clear conversation").click(lambda: [], None, chatbot)

demo.launch()
