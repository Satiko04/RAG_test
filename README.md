# RAG Open Source — Desafio Aula 12
## Stack 100% Gratuita e Local

| Componente | Ferramenta | Substitui |
|---|---|---|
| LLM | **Ollama** (local) | OpenAI GPT |
| Embeddings | **sentence-transformers** | OpenAI Embeddings |
| Banco vetorial | **ChromaDB** local | ChromaDB Cloud |
| Leitura de PDF | **pdfplumber** | — |
| Interface | **Streamlit** | — |

---

## Instalação

### 1. Python e dependências
```bash
pip install -r requirements.txt
```

### 2. Ollama (LLM local)
```bash
# 1. Instale o Ollama: https://ollama.com/download
# 2. Baixe um modelo (escolha conforme sua RAM):
ollama pull phi3:mini      # 4 GB RAM — bom para testes
ollama pull llama3         # 8 GB RAM — recomendado
ollama pull mistral        # 8 GB RAM — alternativa excelente
ollama pull gemma2         # 8 GB RAM — boa opção

# 3. Inicie o servidor
ollama serve
```

---

## Como usar

### Interface Web (recomendado)
```bash
streamlit run interface_web.py
```
Acesse: http://localhost:8501

### Linha de comando
```bash
python rag_app.py
```

### No seu código
```python
from rag_app import RAGOllama

rag = RAGOllama()

# Indexar um arquivo
rag.indexar_arquivo("manual_procedimentos.pdf", estrategia="paragrafos")

# Ou indexar texto diretamente
rag.indexar_texto(meu_texto, fonte="manual.txt", estrategia="paragrafos")

# Consultar
resultado = rag.consultar("Como solicitar férias?")
print(resultado["resposta"])

# Ver fontes usadas
for fonte in resultado["fontes"]:
    print(fonte["texto"][:100])
```

---

## Estratégias de Chunking disponíveis

| Estratégia | Melhor para |
|---|---|
| `paragrafos` | Manuais, relatórios (padrão) |
| `sentencas` | Textos corridos, artigos |
| `fixo` | Textos sem estrutura clara |
| `secoes` | Legislações, normativos (Art., Capítulo...) |

---

## Variáveis de ambiente (opcional)

```bash
export OLLAMA_BASE_URL=http://localhost:11434  # padrão
export OLLAMA_MODEL=llama3                      # modelo do Ollama
export EMBED_MODEL=all-MiniLM-L6-v2            # modelo de embedding
```

---

## Estrutura dos arquivos

```
rag_ollama/
├── rag_app.py        # Pipeline RAG completo
├── interface_web.py  # Interface Streamlit
├── requirements.txt  # Dependências
├── README.md         # Este arquivo
└── chroma_db/        # Banco vetorial (criado automaticamente)
```

---

## Critérios do desafio ✅

- ✅ Documento processado e indexado
- ✅ Chunking adequado implementado (4 estratégias)
- ✅ Busca semântica funcionando (ChromaDB + sentence-transformers)
- ✅ Respostas fundamentadas no contexto (Ollama)
- ✅ Citação de fontes ([Fonte N] inline)
- ✅ Avaliação automatizada com métricas RAG
- ✅ Interface web com Streamlit
- ✅ Suporte a PDF, TXT e texto direto
- ✅ Modo conversacional com reformulação de perguntas
