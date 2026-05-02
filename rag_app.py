"""
Sistema RAG - Desafio da Aula 12
Stack 100% Open Source / Gratuita:
  - Ollama (LLM local): llama3 ou qualquer modelo instalado
  - ChromaDB (banco vetorial): gratuito, roda local
  - sentence-transformers (embeddings): gratuito, roda local
  - PyMuPDF ou pdfplumber (leitura de PDF): gratuito
"""

import os
import re
import json
from typing import List, Optional
from dataclasses import dataclass

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import requests


# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:3.8b")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2") # sentence-transformers grátis
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "manual_procedimentos"


# ─────────────────────────────────────────────
# CLIENTE OLLAMA
# ─────────────────────────────────────────────
def ollama_chat(messages: list, model: str = OLLAMA_MODEL) -> str:
    """Chama o Ollama via biblioteca oficial."""
    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=messages,
        )
        return response.message.content.strip()
    except Exception as e:
        raise RuntimeError(
            f"Erro ao conectar ao Ollama: {e}\n"
            "Verifique se o Ollama está rodando e o modelo está instalado:\n"
            f"ollama pull {model}"
        )

def ollama_available() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False

# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────
class Chunker:
    """Estratégias de chunking — igual ao material da aula."""

    @staticmethod
    def por_tamanho_fixo(texto: str, tamanho: int = 500, overlap: int = 50) -> List[str]:
        chunks, inicio = [], 0
        while inicio < len(texto):
            fim = inicio + tamanho
            chunks.append(texto[inicio:fim].strip())
            inicio = fim - overlap
        return [c for c in chunks if c]

    @staticmethod
    def por_sentencas(texto: str, sentencas_por_chunk: int = 4) -> List[str]:
        sentencas = re.split(r'(?<=[.!?])\s+', texto)
        chunks = []
        for i in range(0, len(sentencas), sentencas_por_chunk):
            chunk = ' '.join(sentencas[i:i + sentencas_por_chunk])
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    @staticmethod
    def por_paragrafos(texto: str) -> List[str]:
        return [p.strip() for p in texto.split('\n\n') if p.strip()]

    @staticmethod
    def por_secoes(texto: str, marcador: str = r'^(?:Art\.|Capítulo|Seção|\d+\.)') -> List[dict]:
        linhas = texto.split('\n')
        secoes, atual = [], {"titulo": "Introdução", "conteudo": []}
        for linha in linhas:
            if re.match(marcador, linha.strip()):
                if atual["conteudo"]:
                    atual["conteudo"] = '\n'.join(atual["conteudo"])
                    secoes.append(atual)
                atual = {"titulo": linha.strip(), "conteudo": []}
            else:
                atual["conteudo"].append(linha)
        if atual["conteudo"]:
            atual["conteudo"] = '\n'.join(atual["conteudo"])
            secoes.append(atual)
        return secoes


# ─────────────────────────────────────────────
# SISTEMA RAG PRINCIPAL
# ─────────────────────────────────────────────
class RAGOllama:
    """
    Pipeline RAG completo usando:
      - Ollama como LLM (local, gratuito)
      - sentence-transformers como embedder (local, gratuito)
      - ChromaDB como banco vetorial (local, gratuito)
    """

    def __init__(self, nome_colecao: str = COLLECTION_NAME):
        # Embedding local via sentence-transformers
        self.embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

        # ChromaDB persistente em disco
        self.chroma = chromadb.PersistentClient(path=CHROMA_PATH)
        self.colecao = self.chroma.get_or_create_collection(
            name=nome_colecao,
            embedding_function=self.embed_fn,
        )
        self.chunker = Chunker()
        self.historico: List[dict] = []

    # ── Indexação ─────────────────────────────
    def indexar_texto(self, texto: str, fonte: str = "documento",
                      estrategia: str = "paragrafos") -> int:
        """Indexa um texto usando a estratégia de chunking escolhida."""
        if estrategia == "fixo":
            chunks = self.chunker.por_tamanho_fixo(texto, 600, 60)
        elif estrategia == "sentencas":
            chunks = self.chunker.por_sentencas(texto, 4)
        elif estrategia == "secoes":
            secoes = self.chunker.por_secoes(texto)
            chunks = [s["conteudo"] for s in secoes if s["conteudo"].strip()]
        else:  # paragrafos (padrão)
            chunks = self.chunker.por_paragrafos(texto)

        inicio = self.colecao.count()
        ids      = [f"chunk_{inicio + i}" for i in range(len(chunks))]
        metadados = [{"fonte": fonte, "chunk_index": i, "estrategia": estrategia}
                     for i in range(len(chunks))]

        self.colecao.add(documents=chunks, metadatas=metadados, ids=ids)
        print(f"✅ {len(chunks)} chunks indexados de '{fonte}'. Total: {self.colecao.count()}")
        return len(chunks)

    def indexar_arquivo(self, caminho: str, estrategia: str = "paragrafos") -> int:
        """Lê .txt ou .pdf e indexa."""
        ext = os.path.splitext(caminho)[1].lower()
        if ext == ".pdf":
            texto = self._ler_pdf(caminho)
        else:
            with open(caminho, encoding="utf-8") as f:
                texto = f.read()
        return self.indexar_texto(texto, fonte=os.path.basename(caminho),
                                  estrategia=estrategia)

    @staticmethod
    def _ler_pdf(caminho: str) -> str:
        """Extrai texto de PDF usando pdfplumber (open source)."""
        try:
            import pdfplumber
            texto = []
            with pdfplumber.open(caminho) as pdf:
                for pagina in pdf.pages:
                    t = pagina.extract_text()
                    if t:
                        texto.append(t)
            return "\n\n".join(texto)
        except ImportError:
            raise RuntimeError("Instale pdfplumber: pip install pdfplumber")

    # ── Recuperação ───────────────────────────
    def recuperar(self, pergunta: str, k: int = 4) -> list:
        resultados = self.colecao.query(query_texts=[pergunta], n_results=k)
        return [
            {"texto": doc, "meta": meta}
            for doc, meta in zip(
                resultados["documents"][0],
                resultados["metadatas"][0],
            )
        ]

    # ── Geração ───────────────────────────────
    def gerar_resposta(self, pergunta: str, contextos: list) -> str:
        contexto_fmt = "\n\n".join(
            f"[Fonte {i+1} — {c['meta'].get('fonte','?')}]:\n{c['texto']}"
            for i, c in enumerate(contextos)
        )
        system = (
            "Você é um assistente especializado que responde perguntas "
            "EXCLUSIVAMENTE com base no contexto fornecido. "
            "Se a informação não estiver no contexto, diga claramente que não encontrou. "
            "Cite as fontes usando [Fonte N] após cada informação usada. "
            "Responda em português."
        )
        user_prompt = f"""CONTEXTO:
{contexto_fmt}

PERGUNTA: {pergunta}

RESPOSTA (cite as fontes):"""

        return ollama_chat([
            {"role": "system", "content": system},
            {"role": "user",   "content": user_prompt},
        ])

    # ── Reformulação conversacional ───────────
    def reformular_pergunta(self, pergunta: str) -> str:
        if not self.historico:
            return pergunta
        hist_txt = "\n".join(
            f"{'Usuário' if m['role']=='user' else 'Assistente'}: {m['content']}"
            for m in self.historico[-4:]
        )
        prompt = (
            f"Dado o histórico abaixo, reformule a pergunta para ser autocontida.\n\n"
            f"HISTÓRICO:\n{hist_txt}\n\n"
            f"PERGUNTA ORIGINAL: {pergunta}\n\n"
            f"PERGUNTA REFORMULADA (apenas a pergunta, sem explicação):"
        )
        return ollama_chat([{"role": "user", "content": prompt}])

    # ── Consulta completa ─────────────────────
    def consultar(self, pergunta: str, k: int = 4, conversacional: bool = True) -> dict:
        pergunta_final = self.reformular_pergunta(pergunta) if conversacional else pergunta
        contextos = self.recuperar(pergunta_final, k)
        resposta  = self.gerar_resposta(pergunta_final, contextos)

        # Atualizar histórico
        self.historico.append({"role": "user",      "content": pergunta})
        self.historico.append({"role": "assistant",  "content": resposta})

        return {
            "pergunta_original":   pergunta,
            "pergunta_reformulada": pergunta_final,
            "resposta":            resposta,
            "fontes":              contextos,
        }

    def limpar_historico(self):
        self.historico = []

    def limpar_banco(self):
        """Apaga todos os documentos do banco vetorial."""
        nome = self.colecao.name
        self.chroma.delete_collection(nome)
        self.colecao = self.chroma.get_or_create_collection(
            name=nome,
            embedding_function=self.embed_fn,
        )
        print("🗑️ Banco vetorial limpo.")

    def listar_fontes(self) -> list:
        """Retorna lista de documentos únicos indexados no banco."""
        if self.colecao.count() == 0:
            return []
        resultados = self.colecao.get(include=["metadatas"])
        fontes = list({
            m.get("fonte", "desconhecido")
            for m in resultados["metadatas"]
            if m
        })
        return sorted(fontes)

    def stats(self) -> dict:
        return {"total_chunks": self.colecao.count(), "modelo_llm": OLLAMA_MODEL,
                "modelo_embed": EMBED_MODEL}


# ─────────────────────────────────────────────
# AVALIAÇÃO
# ─────────────────────────────────────────────
def avaliar_resposta(pergunta: str, resposta: str, contexto: str,
                     resposta_esperada: str = None) -> dict:
    """Avalia a resposta usando o próprio Ollama como juiz."""
    ref = f"\nRESPOSTA ESPERADA: {resposta_esperada}" if resposta_esperada else ""
    prompt = f"""Avalie a resposta de um sistema RAG e retorne APENAS um JSON válido.

PERGUNTA: {pergunta}
CONTEXTO: {contexto[:800]}
RESPOSTA GERADA: {resposta}{ref}

Retorne exatamente este JSON (valores entre 0.0 e 1.0):
{{"relevancia": 0.0, "faithfulness": 0.0, "completude": 0.0, "clareza": 0.0, "comentarios": ""}}"""

    raw = ollama_chat([{"role": "user", "content": prompt}])
    try:
        # Extrai JSON mesmo que o modelo adicione texto ao redor
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        return json.loads(match.group()) if match else {"erro": raw}
    except Exception:
        return {"erro": raw}


# ─────────────────────────────────────────────
# DEMO CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Open Source — Ollama + ChromaDB + SentenceTransformers")
    print("=" * 60)

    if not ollama_available():
        print("\n⚠️  Ollama não detectado em", OLLAMA_BASE_URL)
        print("   Instale em https://ollama.com e execute: ollama pull llama3")
        exit(1)

    rag = RAGOllama()

    # Documento de exemplo (manual de procedimentos)
    manual = """
Política de Férias
Para solicitar férias, o colaborador deve preencher o formulário F-001 com 30 dias de antecedência mínima. O período de férias deve ser acordado com o gestor direto. Férias podem ser parceladas em até 3 períodos, sendo o menor deles de 5 dias corridos.

Jornada de Trabalho
O horário de expediente é das 8h às 18h, com 1 hora de almoço. Existe tolerância de 15 minutos para entrada e saída. Trabalho remoto pode ser solicitado mediante aprovação do gestor, limitado a 2 dias por semana.

Atestados Médicos
Atestado médico de até 2 dias: entregar ao RH em 48 horas. Atestado superior a 3 dias: enviar digitalizado ao RH em até 48 horas e apresentar original em até 5 dias úteis. Atestados de acompanhamento de familiar são aceitos para até 1 dia por mês.

Promoções e Carreira
A promoção por mérito ocorre a cada 2 anos, vinculada à avaliação de desempenho semestral. A nota mínima para elegibilidade é 7,0. Promoções por indicação do gestor podem ocorrer fora do ciclo mediante aprovação do Comitê de RH.

Equipamentos e TI
Todo colaborador recebe notebook corporativo no primeiro dia. Solicitações de software devem ser feitas via portal TI com 3 dias úteis de antecedência. Danos ao equipamento por negligência podem resultar em desconto em folha conforme política patrimonial.

Benefícios
Vale-refeição: R$ 35,00 por dia útil trabalhado. Plano de saúde: cobertura nacional, titular + 2 dependentes sem custo adicional. Auxílio academia: reembolso de até R$ 150/mês mediante comprovante. Seguro de vida: 24x o salário bruto.
"""

    print("\n📄 Indexando manual de procedimentos...")
    rag.indexar_texto(manual, fonte="manual_rh_v1.pdf", estrategia="paragrafos")

    perguntas_demo = [
        "Como faço para solicitar férias?",
        "Qual é o horário de trabalho e posso fazer home office?",
        "Quais são os benefícios de saúde oferecidos?",
        "Como funciona a promoção por mérito?",
    ]

    print("\n" + "─" * 60)
    for pergunta in perguntas_demo:
        print(f"\n❓ {pergunta}")
        resultado = rag.consultar(pergunta, conversacional=False)
        print(f"💬 {resultado['resposta']}")
        print("─" * 60)

    print("\n✅ Demo concluída! Use a classe RAGOllama no seu projeto.")
    print(f"📊 Stats: {rag.stats()}")
