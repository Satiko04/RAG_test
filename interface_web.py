"""
Interface Web para o Sistema RAG
Roda com: streamlit run interface_web.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from rag_app import RAGOllama, avaliar_resposta, ollama_available, OLLAMA_MODEL, EMBED_MODEL

# ── Config da página ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Open Source",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilo ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  .stApp { background: #0d1117; color: #e6edf3; }
  .main-header { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem;
    color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: .5rem; }
  .source-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 12px; margin: 6px 0; font-size: .85rem; color: #8b949e; }
  .source-card strong { color: #58a6ff; }
  .metric-card { background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 16px; text-align: center; }
  .metric-val { font-size: 2rem; font-weight: 600; color: #3fb950; }
  .badge { display:inline-block; background:#1f6feb33; color:#58a6ff;
    border:1px solid #1f6feb; border-radius:4px; padding:2px 8px;
    font-family:'IBM Plex Mono',monospace; font-size:.75rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = RAGOllama()
if "historico_chat" not in st.session_state:
    st.session_state.historico_chat = []
if "avaliacao" not in st.session_state:
    st.session_state.avaliacao = None

rag: RAGOllama = st.session_state.rag

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuração")

    status_ollama = "🟢 Conectado" if ollama_available() else "🔴 Offline"
    st.markdown(f"**Ollama:** {status_ollama}")
    st.markdown(f'<span class="badge">{OLLAMA_MODEL}</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="badge">{EMBED_MODEL}</span>', unsafe_allow_html=True)

    stats = rag.stats()
    st.metric("Chunks indexados", stats["total_chunks"])

    st.divider()
    st.markdown("### 📄 Indexar Documento")

    estrategia = st.selectbox(
        "Estratégia de chunking",
        ["paragrafos", "sentencas", "fixo", "secoes"],
        help="paragrafos=melhor para manuais | sentencas=textos corridos | secoes=documentos normativos"
    )

    modo_indexacao = st.radio(
        "Ao indexar novo documento:",
        ["➕ Adicionar ao banco", "🔄 Substituir banco (limpar antes)"],
        help="'Substituir' apaga documentos anteriores — evita mistura de conteúdo"
    )
    limpar_antes = modo_indexacao.startswith("🔄")

    uploaded = st.file_uploader("Upload (.txt ou .pdf)", type=["txt", "pdf"])
    if uploaded and st.button("📥 Indexar arquivo"):
        tmp_dir = os.environ.get("TEMP", os.environ.get("TMP", "/tmp"))
        caminho_tmp = os.path.join(tmp_dir, uploaded.name)
        with open(caminho_tmp, "wb") as f:
            f.write(uploaded.read())
        with st.spinner("Indexando..."):
            if limpar_antes:
                rag.limpar_banco()
                st.info("🗑️ Banco limpo! Indexando novo documento...")
            n = rag.indexar_arquivo(caminho_tmp, estrategia=estrategia)
        st.success(f"✅ {n} chunks indexados de '{uploaded.name}'!")
        st.rerun()

    st.divider()
    st.markdown("### 📝 Texto direto")
    texto_manual = st.text_area("Cole o texto aqui", height=150,
        placeholder="Cole o conteúdo do manual de procedimentos...")
    fonte_nome = st.text_input("Nome da fonte", "manual.txt")
    if st.button("📥 Indexar texto") and texto_manual.strip():
        with st.spinner("Indexando..."):
            if limpar_antes:
                rag.limpar_banco()
            n = rag.indexar_texto(texto_manual, fonte=fonte_nome, estrategia=estrategia)
        st.success(f"✅ {n} chunks indexados!")
        st.rerun()

    st.divider()
    st.markdown("### 📂 Documentos no banco")
    fontes = rag.listar_fontes()
    if fontes:
        for fonte in fontes:
            st.markdown(f"- 📄 `{fonte}`")
    else:
        st.caption("Nenhum documento indexado.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Limpar conversa"):
            rag.limpar_historico()
            st.session_state.historico_chat = []
            st.success("Histórico limpo!")
    with col2:
        if st.button("💣 Limpar banco", type="primary"):
            rag.limpar_banco()
            rag.limpar_historico()
            st.session_state.historico_chat = []
            st.success("Banco e histórico limpos!")
            st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🔍 RAG Open Source</p>', unsafe_allow_html=True)
st.caption("Ollama · ChromaDB · SentenceTransformers — 100% local e gratuito")

if stats["total_chunks"] == 0:
    st.info("👈 Nenhum documento indexado ainda. Use a barra lateral para adicionar documentos.")

tab_chat, tab_eval, tab_guide = st.tabs(["💬 Chat", "📊 Avaliação", "📘 Guia"])

# ── Tab Chat ──────────────────────────────────────────────────────────────────
with tab_chat:
    # Exibir histórico
    for msg in st.session_state.historico_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("fontes"):
                with st.expander(f"📎 {len(msg['fontes'])} fonte(s) utilizadas"):
                    for i, f in enumerate(msg["fontes"]):
                        st.markdown(
                            f'<div class="source-card"><strong>[{i+1}] '
                            f'{f["meta"].get("fonte","N/A")}</strong> — '
                            f'chunk #{f["meta"].get("chunk_index","?")}<br>'
                            f'{f["texto"][:300]}{"..." if len(f["texto"])>300 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Input
    if pergunta := st.chat_input("Faça uma pergunta sobre os documentos indexados..."):
        if stats["total_chunks"] == 0:
            st.error("Indexe algum documento primeiro!")
        elif not ollama_available():
            st.error("Ollama não está rodando. Execute `ollama serve` no terminal.")
        else:
            st.session_state.historico_chat.append({"role": "user", "content": pergunta})
            with st.chat_message("user"):
                st.markdown(pergunta)

            with st.chat_message("assistant"):
                with st.spinner("Buscando e gerando resposta..."):
                    resultado = rag.consultar(pergunta)

                st.markdown(resultado["resposta"])

                if resultado["pergunta_reformulada"] != resultado["pergunta_original"]:
                    st.caption(f"🔄 Pergunta reformulada: *{resultado['pergunta_reformulada']}*")

                with st.expander(f"📎 {len(resultado['fontes'])} fonte(s) utilizadas"):
                    for i, f in enumerate(resultado["fontes"]):
                        st.markdown(
                            f'<div class="source-card"><strong>[{i+1}] '
                            f'{f["meta"].get("fonte","N/A")}</strong> — '
                            f'chunk #{f["meta"].get("chunk_index","?")}<br>'
                            f'{f["texto"][:300]}{"..." if len(f["texto"])>300 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                st.session_state.historico_chat.append({
                    "role": "assistant",
                    "content": resultado["resposta"],
                    "fontes": resultado["fontes"],
                })
                st.session_state.avaliacao = resultado

# ── Tab Avaliação ─────────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("### 📊 Avaliar última resposta")
    st.caption("O próprio Ollama age como juiz avaliando a qualidade da resposta RAG.")

    if not st.session_state.avaliacao:
        st.info("Faça uma pergunta na aba Chat para avaliar.")
    else:
        res = st.session_state.avaliacao
        st.markdown(f"**Pergunta:** {res['pergunta_original']}")
        st.markdown(f"**Resposta:** {res['resposta'][:400]}...")

        resposta_esp = st.text_area("Resposta esperada (opcional — para comparação):", height=80)

        if st.button("🔎 Avaliar com Ollama"):
            contexto_txt = "\n".join(c["texto"] for c in res["fontes"])
            with st.spinner("Avaliando..."):
                metricas = avaliar_resposta(
                    res["pergunta_original"], res["resposta"],
                    contexto_txt, resposta_esp or None,
                )

            if "erro" in metricas:
                st.error(f"Erro na avaliação: {metricas['erro']}")
            else:
                cols = st.columns(4)
                nomes = ["relevancia", "faithfulness", "completude", "clareza"]
                emojis = ["🎯", "🔒", "📋", "✨"]
                for col, nome, emoji in zip(cols, nomes, emojis):
                    val = metricas.get(nome, 0)
                    with col:
                        st.markdown(
                            f'<div class="metric-card">{emoji}<br>'
                            f'<span class="metric-val">{val:.0%}</span><br>'
                            f'<small>{nome.capitalize()}</small></div>',
                            unsafe_allow_html=True,
                        )
                if metricas.get("comentarios"):
                    st.info(f"💬 {metricas['comentarios']}")

# ── Tab Guia ──────────────────────────────────────────────────────────────────
with tab_guide:
    st.markdown("""
### 🚀 Como executar este projeto

#### 1. Instalar dependências
```bash
pip install chromadb sentence-transformers requests streamlit pdfplumber
```

#### 2. Instalar e iniciar o Ollama
```bash
# Instale em https://ollama.com
ollama pull llama3        # ou: mistral, phi3, gemma2, qwen2
ollama serve              # inicia o servidor na porta 11434
```

#### 3. Rodar a interface web
```bash
streamlit run interface_web.py
```

#### 4. Usar apenas via código
```python
from rag_app import RAGOllama

rag = RAGOllama()
rag.indexar_texto("Texto do manual...", fonte="manual.pdf")
resultado = rag.consultar("Como solicitar férias?")
print(resultado["resposta"])
```

---

### 🆓 Stack 100% Gratuita

| Componente | Ferramenta | Custo |
|---|---|---|
| LLM | Ollama (local) | Gratuito |
| Embeddings | sentence-transformers | Gratuito |
| Banco vetorial | ChromaDB (local) | Gratuito |
| Leitura de PDF | pdfplumber | Gratuito |
| Interface | Streamlit | Gratuito |

### 🤖 Modelos recomendados para o Ollama

| Modelo | RAM necessária | Qualidade |
|---|---|---|
| `phi3:mini` | 4 GB | Boa para testes |
| `llama3` | 8 GB | Excelente |
| `mistral` | 8 GB | Excelente |
| `gemma2` | 8 GB | Muito boa |
| `qwen2.5` | 8 GB | Muito boa |
| `llama3:70b` | 40 GB | Excepcional |
""")
