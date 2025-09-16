import base64
import json
import os
import requests
import uuid
import re
import contextlib
from dataclasses import dataclass, asdict, fields,field
from datetime import datetime, time
from io import BytesIO
from typing import Any, Dict, List
import streamlit as st

# Verificação de dependências para PDF
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    pdfplumber = None
    HAS_PDF = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

PREFS_FILE = "viasell_task_prefs.json"

# Importar módulos existentes
from configuracao_teamwork import TEAMWORK_CONFIG
from ml_utils import MLAnalyzer
from teamwork_client import TeamworkClient

# =========================
#  Logs de lançamentos (sem BD)
# =========================
LOG_FILE = "viasell_lancamentos_log.json"  # JSON-L (1 registro por linha). Lemos array JSON também.

def _append_log(entry: dict, log_file: str = LOG_FILE) -> None:
    """Acrescenta 1 registro ao arquivo de log (NDJSON)."""
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def _read_logs(log_file: str = LOG_FILE) -> List[dict]:
    """Lê histórico do LOG_FILE. Aceita NDJSON e também JSON array."""
    if not os.path.exists(log_file):
        return []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return []
        # modo array
        if content.startswith("["):
            data = json.loads(content)
            return data if isinstance(data, list) else [data]
        # modo NDJSON (1 linha = 1 JSON)
        out = []
        for ln in content.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                # ignora linhas quebradas
                pass
        return out
    except Exception:
        return []

def _export_logs_json_array(logs: List[dict]) -> str:
    """Exporta lista como JSON (array) bonito."""
    return json.dumps(logs, ensure_ascii=False, indent=2)


# =========================
#  Util / Namespace p/ keys
# =========================
NVF_NS = "nvf"

def K(name: str) -> str:
    return f"{NVF_NS}:{name}"

@contextlib.contextmanager
def _mute_debug_ui():
    """Desliga write/info/success/warning temporariamente."""
    _w, _i, _s, _wa = st.write, st.info, st.success, st.warning
    try:
        st.write   = lambda *a, **k: None
        st.info    = lambda *a, **k: None
        st.success = lambda *a, **k: None
        st.warning = lambda *a, **k: None
        yield
    finally:
        st.write, st.info, st.success, st.warning = _w, _i, _s, _wa

# =========================
#  Modelos de dados
# =========================
@dataclass
class RegistroAtividade:
    """Classe para representar um registro de atividade"""
    id: str
    data: str                  # dd/mm/YYYY
    executor: str
    hr_inicio: str             # HH:MM:SS
    hr_fim: str                # HH:MM:SS
    descricao: str
    total_hrs: str             # HH:MM:SS
    faturavel: bool
    tarefa_id: str = ""
    tarefa_nome: str = ""
    executor_id: str = ""      # id da pessoa no Teamwork (se selecionado)
    ml_sugestao: str = ""
    ml_confianca: float = 0.0

@dataclass
class FichaServico:
    """Ficha de serviço completa"""
    id: str
    cliente: str
    projeto_id: str
    vertical: str
    tipo_servico: str
    valor_hora: float
    registros: List[RegistroAtividade]
    data_criacao: str
    status: str = "Em Andamento"

@dataclass
class ConfiguracaoViasell:
    tag_teamwork: str = "apontavel"
    area_projeto: str = "Implantação"
    valor_hora_padrao: float = 180.0
    vertical_padrao: str = "Construshow"
    exigir_consultor: bool = True
    ocultar_painel_analise: bool = True
    # NOVO: mapa projeto -> {task_id: task_name}
    tarefas_por_projeto: Dict[str, Dict[str, str]] = field(default_factory=dict)

# =========================
#  Gerenciador de Configurações
# =========================
class ConfigManager:
    """Gerenciador de configurações persistentes"""
    
    def __init__(self):
        self.config_file = "config_viasell.json"
        self.config = self.carregar_config()
    
    def carregar_config(self) -> ConfiguracaoViasell:
        """Carrega configurações do arquivo"""
        if not os.path.exists(self.config_file):
            return ConfiguracaoViasell()
        
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Filtrar apenas campos válidos
            config_data = _filter_keys(data, ConfiguracaoViasell)
            return ConfiguracaoViasell(**config_data)
        
        except Exception as e:
            st.error(f"Erro ao carregar configurações: {e}")
            return ConfiguracaoViasell()
    
    def salvar_config(self):
        """Salva configurações no arquivo"""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"Erro ao salvar configurações: {e}")

# =========================
#  Helpers genéricos
# =========================
def _filter_keys(d: dict, data_cls):
    """Retorna apenas as chaves de d que existem como campos do dataclass data_cls."""
    allowed = {f.name for f in fields(data_cls)}
    return {k: v for k, v in (d or {}).items() if k in allowed}

def _safe_decode(b: bytes) -> str:
    """Decodifica bytes de forma segura"""
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return b.decode("latin-1", errors="ignore")

# =========================
#  Teamwork API - Com Fallbacks
# =========================
def _teamwork_auth_headers():
    token = base64.b64encode(f"{TEAMWORK_CONFIG['api_key']}:x".encode()).decode()
    return {"Authorization": f"Basic {token}", "Accept": "application/json"}

def get_projects_teamwork() -> List[Dict[str, Any]]:
    """Busca projetos do Teamwork usando API v1 (fallback confiável)"""
    base = TEAMWORK_CONFIG['base_url'].rstrip('/')
    url = f"{base}/projects.json"
    
    try:
        resp = requests.get(url, headers=_teamwork_auth_headers(), timeout=20)
        resp.raise_for_status()
        data = resp.json()
        
        projects = data.get("projects", [])
        project_list = []
        
        for proj in projects:
            project_info = {
                "id": str(proj.get("id", "")),
                "name": proj.get("name", ""),
                "category": proj.get("category", {}).get("name", "") if proj.get("category") else "",
                "status": proj.get("status", ""),
                "company": proj.get("company", {}).get("name", "") if proj.get("company") else "",
                "description": proj.get("description", "")[:100] + "..." if len(proj.get("description", "")) > 100 else proj.get("description", "")
            }
            project_list.append(project_info)
        
        # Ordenar por nome
        project_list.sort(key=lambda x: x["name"].lower())
        return project_list
        
    except Exception as e:
        st.error(f"Erro ao buscar projetos do Teamwork: {e}")
        return []

def get_tasks_by_tag_and_project(
    project_id: str,
    tag_query: str = "",
    inherit_from_parent: bool = True,
    include_completed: bool = True,
) -> List[Dict[str, Any]]:
    """
    Busca tarefas (e subtarefas) de um projeto.
    - tag_query: se informado, retorna apenas tarefas que tenham a tag (case-insensitive).
    - inherit_from_parent=True: subtarefas herdarem tags do pai (útil quando só o pai tem a tag).
    - include_completed=True: inclui tarefas concluídas.
    Retorna: [{"id": "...", "name": "...", "tags": ["...","..."], "parentId": "..."}]
    """
    base = TEAMWORK_CONFIG['base_url'].rstrip('/')

    def _try_fetch(params: Dict[str, str]) -> List[dict]:
        items_all = []
        page = 1
        while True:
            q = params.copy()
            q["page"] = str(page)
            url = f"{base}/projects/{project_id}/tasks.json"
            try:
                r = requests.get(url, headers=_teamwork_auth_headers(), params=q, timeout=30)
                r.raise_for_status()
                data = r.json() or {}
            except Exception:
                # fallback minimalista
                if "include" in q: q.pop("include", None)
                if "nestSubTasks" in q: q.pop("nestSubTasks", None)
                if "status" in q: q.pop("status", None)
                if "includeCompletedTasks" in q: q.pop("includeCompletedTasks", None)
                try:
                    r = requests.get(url, headers=_teamwork_auth_headers(), params=q, timeout=30)
                    r.raise_for_status()
                    data = r.json() or {}
                except Exception:
                    return []

            items = data.get("tasks") or data.get("todo-items") or []
            items_all.extend(items)

            # Teamwork v1 não dá um "hasMore" padronizado; paramos se veio menos que o pageSize
            page_size = int(q.get("pageSize", 200))
            if len(items) < page_size:
                break
            page += 1
        return items_all

    # primeira tentativa: com tags + subtarefas + completos
    params_pref = {
        "include": "tags,subTasks",
        "nestSubTasks": "true",
        "pageSize": "500",
        "status": "all" if include_completed else "active",
        "includeCompletedTasks": "true" if include_completed else "false",
    }
    items = _try_fetch(params_pref)
    if not items:
        # segunda: mais simples
        items = _try_fetch({"include": "tags", "pageSize": "200"})

    # normalizar + achatar subtarefas (se vierem aninhadas)
    def _collect(item, parent_id=None, parent_tags=None, out=None):
        if out is None: out = []
        name = item.get("content") or item.get("name") or ""
        tid = str(item.get("id") or item.get("id_str") or item.get("taskId") or "")
        tags = [t.get("name","") for t in (item.get("tags") or []) if t.get("name")]
        # herança
        if inherit_from_parent and parent_tags:
            for t in parent_tags:
                if t not in tags:
                    tags.append(t)
        out.append({
            "id": tid,
            "name": name,
            "tags": tags,
            "parentId": parent_id or str(item.get("parentTaskId") or ""),
        })
        # subtarefas podem vir em "subTasks"
        for sub in item.get("subTasks") or []:
            _collect(sub, parent_id=tid, parent_tags=tags, out=out)
        return out

    flat: List[Dict[str, Any]] = []
    for it in items:
        _collect(it, out=flat)

    # filtro por tag (case-insensitive) se solicitado
    tag_q = (tag_query or "").strip().lower()
    if tag_q:
        flat = [t for t in flat if any(tag_q == (tg or "").lower() for tg in (t.get("tags") or []))]

    # remover duplicatas (pode acontecer quando a API devolve pai + filhos em diferentes páginas)
    seen = set()
    dedup = []
    for t in flat:
        if t["id"] and t["id"] not in seen:
            seen.add(t["id"])
            dedup.append(t)

    # ordenar por nome
    dedup.sort(key=lambda x: (x.get("name") or "").lower())
    return dedup

# =========================
#  Utilitários de PDF e Texto - CORRIGIDOS PARA VIASELL
# =========================
def extract_text_from_upload(file) -> str:
    """Extrai texto de PDF (pdfplumber) ou TXT. Retorna string limpa."""
    name = (getattr(file, "name", "") or "").lower()
    
    if name.endswith(".pdf"):
        if not HAS_PDF:
            raise RuntimeError("Para ler PDF, instale 'pdfplumber' (pip install pdfplumber).")
        
        text_parts = []
        with pdfplumber.open(file) as pdf:
            for pg in pdf.pages:
                t = pg.extract_text() or ""
                text_parts.append(t)
        return "\n".join(text_parts)
    
    elif name.endswith(".txt"):
        data = file.read()
        return _safe_decode(data)
    
    elif name.endswith(".json"):
        data = file.read()
        return _safe_decode(data)
    
    else:
        data = file.read()
        return _safe_decode(data)

def extrair_dados_viasell_corrigido(texto: str) -> dict:
    """Extrai dados da ficha Viasell trazendo a DESCRIÇÃO da seção 'Serviço Exec.' para cada registro
       e prefixando com o número da Ficha quando disponível."""
    dados = {
        'cliente': 'Cliente não identificado',
        'projeto_id': '',
        'vertical': 'Construshow',
        'tipo_servico': 'Implantação',
        'valor_hora': 180.0,
        'registros': []
    }

    st.write("**🔍 Analisando texto extraído:**")
    linhas = texto.split('\n')
    st.write(f"Total de linhas: {len(linhas)}")

    # -------- Número da Ficha (para prefixo) --------
    ficha_num = extrair_numero_ficha(texto)
    ficha_prefix = f"FICHA {ficha_num} - " if ficha_num else ""
    if ficha_num:
        st.info(f"🧾 Número da Ficha detectado: {ficha_num}")

    # ===== Dados básicos =====
    cliente_patterns = [
        r'Cliente\s*\n?\s*(\d+\s*-\s*[^\n\r]+)',
        r'(\d{4,6}\s*-\s*[A-Za-z][^0-9\n\r]+)',
        r'Cliente[:\s]*([^\n\r]+)'
    ]
    for pattern in cliente_patterns:
        match = re.search(pattern, texto, re.IGNORECASE | re.MULTILINE)
        if match:
            dados['cliente'] = match.group(1).strip()
            break

    m_proj = re.search(r'(\d{4,6})', dados['cliente'])
    if m_proj:
        dados['projeto_id'] = m_proj.group(1)

    tipo_patterns = [
        r'(Implantação|Personalização|Serviços|Deslocamento)\s*-\s*[A-Z]+',
        r'Ficha:\s*\d+\s*\n?\s*(Implantação|Personalização|Serviços|Deslocamento)'
    ]
    for pattern in tipo_patterns:
        match = re.search(pattern, texto, re.IGNORECASE)
        if match:
            dados['tipo_servico'] = match.group(1)
            break

    valor_patterns = [
        r'Valor/Hr\s+Técnica\s+R\$\s*(\d+[.,]?\d*)',
        r'(\d+[.,]\d{2})\s*(?=.*valor.*hora)',
        r'R\$\s*(\d+[.,]?\d*)'
    ]
    for pattern in valor_patterns:
        match = re.search(pattern, texto, re.IGNORECASE)
        if match:
            try:
                dados['valor_hora'] = float(match.group(1).replace(',', '.'))
                break
            except ValueError:
                pass

    # ===== Registros (linha a linha + Serviço Exec.) =====
    st.write("**📋 Registros coletados...**")
    registros: List[Dict[str, Any]] = []

    regex_registro = re.compile(
        r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s+'      # data
        r'(\d+\s*-\s*[^\d]+?)\s+'                    # executor (com código)
        r'(\d{1,2}:\d{2}:\d{2})\s+'                  # início
        r'(\d{1,2}:\d{2}:\d{2})\s+'                  # fim
        r'(\d{1,2}:\d{2}:\d{2})\s+'                  # total
        r'(Sim|Não)',                                 # faturável
        re.IGNORECASE
    )

    encontrados = 0
    for idx, linha in enumerate(linhas):
        m = regex_registro.search(linha or "")
        if not m:
            continue

        data, executor, hr_inicio, hr_fim, total_hrs, cobrar = m.groups()
        executor_limpo = re.sub(r'^\d+\s*-\s*', '', (executor or '').strip())
        faturavel = (cobrar or '').strip().lower() == 'sim'
        data_normalizada = normalizar_data(data)

        # Descrição: pega o bloco "Serviço Exec." relativo a este registro
        desc_exec = buscar_descricao_serv_exec(linhas, idx)
        if not desc_exec:
            desc_exec = f"Atividade executada em {data_normalizada} por {executor_limpo}."

        descricao_final = f"{ficha_prefix}{desc_exec}"

        registro = {
            'data': data_normalizada,
            'executor': executor_limpo,
            'hr_inicio': hr_inicio,
            'hr_fim': hr_fim,
            'total_hrs': total_hrs,
            'descricao': descricao_final,
            'faturavel': faturavel,
            'tarefa_id': '',
            'tarefa_nome': ''
        }
        registros.append(registro)
        encontrados += 1

        st.write(f"✅ Registro {encontrados}: {data_normalizada} - {executor_limpo} ({total_hrs})")
        st.success(f"   📝 {descricao_final[:200]}{'...' if len(descricao_final) > 200 else ''}")

    # ===== Fallback 1: formato flexível =====
    if not registros:
        st.warning("⚠️ Nenhum registro no formato principal; aplicando estratégia alternativa.")
        regex_flex = re.compile(
            r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}).*?(\d{1,2}:\d{2}:\d{2}).*?(\d{1,2}:\d{2}:\d{2}).*?(\d{1,2}:\d{2}:\d{2})'
        )
        for idx, linha in enumerate(linhas):
            mf = regex_flex.search(linha or "")
            if not mf:
                continue

            data, hr_inicio, hr_fim, total_hrs = mf.groups()
            ex_match = re.search(r'(\d+\s*-\s*[A-Za-z][^0-9]+)', linha)
            executor = ex_match.group(1) if ex_match else "Executor não identificado"
            executor_limpo = re.sub(r'^\d+\s*-\s*', '', executor.strip())
            cobrar_match = re.search(r'\b(Sim|Não)\b', linha, re.IGNORECASE)
            faturavel = bool(cobrar_match and cobrar_match.group(1).lower() == 'sim')
            data_normalizada = normalizar_data(data)

            desc_exec = buscar_descricao_serv_exec(linhas, idx) or f"Atividade executada em {data_normalizada} por {executor_limpo}."
            descricao_final = f"{ficha_prefix}{desc_exec}"

            registros.append({
                'data': data_normalizada,
                'executor': executor_limpo,
                'hr_inicio': hr_inicio,
                'hr_fim': hr_fim,
                'total_hrs': total_hrs,
                'descricao': descricao_final,
                'faturavel': faturavel,
                'tarefa_id': '',
                'tarefa_nome': ''
            })

    # ===== Fallback 2: análise manual =====
    if not registros:
        st.write("**🔍 Análise manual das linhas...**")
        for i, linha in enumerate(linhas):
            if not linha.strip():
                continue
            if (re.search(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', linha) and re.search(r'\d{1,2}:\d{2}', linha)):
                st.write(f"**Linha {i+1}:** {linha[:200]}...")
                datas = re.findall(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', linha)
                horarios = re.findall(r'\d{1,2}:\d{2}:\d{2}', linha)

                if datas and len(horarios) >= 2:
                    data = datas[0]
                    hr_inicio = horarios[0]
                    hr_fim = horarios[1] if len(horarios) > 1 else "00:00:00"
                    total_hrs = horarios[2] if len(horarios) > 2 else calcular_total_horas(hr_inicio[:5], hr_fim[:5])

                    ex_match = re.search(r'(\d+\s*-\s*[A-Za-z][^0-9\n\r]+)', linha)
                    executor = "Executor não identificado"
                    if ex_match:
                        executor = re.sub(r'^\d+\s*-\s*', '', ex_match.group(1).strip())

                    faturavel = 'sim' in linha.lower()
                    data_normalizada = normalizar_data(data)

                    desc_exec = buscar_descricao_serv_exec(linhas, i) or f"Atividade executada em {data_normalizada} por {executor}."
                    descricao_final = f"{ficha_prefix}{desc_exec}"

                    registro = {
                        'data': data_normalizada,
                        'executor': executor,
                        'hr_inicio': hr_inicio,
                        'hr_fim': hr_fim,
                        'total_hrs': total_hrs,
                        'descricao': descricao_final,
                        'faturavel': faturavel,
                        'tarefa_id': '',
                        'tarefa_nome': ''
                    }
                    registros.append(registro)
                    st.success(f"✅ Extraído: {data_normalizada} - {executor} ({total_hrs})")

    dados['registros'] = registros
    st.write(f"**📊 Total de registros extraídos: {len(registros)}**")
    return dados

def normalizar_data(data_str: str) -> str:
    """Normaliza formato de data para dd/mm/yyyy"""
    try:
        # Remove espaços e substitui separadores
        data_clean = data_str.strip().replace('-', '/').replace('.', '/')
        
        # Tentar diferentes formatos
        parts = data_clean.split('/')
        if len(parts) == 3:
            dia, mes, ano = parts
            # Se ano com 2 dígitos, assumir 20xx
            if len(ano) == 2:
                ano = f"20{ano}"
            return f"{dia.zfill(2)}/{mes.zfill(2)}/{ano}"
    except:
        pass
    
    return data_str

def debug_texto_extraido(texto: str, max_chars: int = 3000):
    """Função para debug do texto extraído"""
    with st.expander("🔍 Estrutura do Texto Extraído"):
        # Mostrar preview do texto
        preview = texto[:max_chars]
        if len(texto) > max_chars:
            preview += f"\n\n... (texto truncado, total: {len(texto)} caracteres)"
        
        # Dividir em seções para facilitar análise
        st.subheader("📄 Conteúdo Extraído")
        st.code(preview)
        
        # Estatísticas básicas
        linhas = texto.split('\n')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Linhas", len(linhas))
        with col2:
            st.metric("Caracteres", len(texto))
        with col3:
            datas = re.findall(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', texto)
            st.metric("Datas", len(datas))
        with col4:
            horarios = re.findall(r'\d{1,2}:\d{2}', texto)
            st.metric("Horários", len(horarios))
        
        # Mostrar padrões encontrados
        st.subheader("🔍 Padrões Identificados")
        
        # Datas encontradas
        if datas:
            st.write("**📅 Datas encontradas:**")
            for data in datas:
                st.write(f"- {data}")
        
        # Executores encontrados
        executores = re.findall(r'\d+\s*-\s*[A-Za-z][^0-9\n\r]+', texto)
        if executores:
            st.write("**👤 Executores encontrados:**")
            for executor in executores[:5]:  # Mostrar apenas os primeiros 5
                st.write(f"- {executor.strip()}")
        
        # Horários encontrados
        if horarios:
            st.write(f"**🕐 Horários encontrados:** {horarios[:10]}")

# =========================
#  PDF Generator (mantém o mesmo)
# =========================
class PDFGenerator:
    """Gerador de PDF em layout limpo para a ficha Viasoft (com minutos p/ Teamwork)."""

    @staticmethod
    def _hms_to_decimal(hms: str) -> float:
        if not hms:
            return 0.0
        parts = [int(p) for p in hms.split(":")]
        if len(parts) == 2:
            h, m = parts; s = 0
        elif len(parts) == 3:
            h, m, s = parts
        else:
            return 0.0
        return h + m/60.0 + s/3600.0

    @staticmethod
    def _hms_to_minutes(hms: str) -> int:
        if not hms:
            return 0
        parts = [int(p) for p in hms.split(":")]
        if len(parts) == 2:
            h, m = parts; s = 0
        elif len(parts) == 3:
            h, m, s = parts
        else:
            return 0
        total_seconds = h*3600 + m*60 + s
        return int(round(total_seconds/60.0))

    @staticmethod
    def _decimal_to_hms(hours: float) -> str:
        total = int(round(hours * 3600))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _fmt_money(valor: float) -> str:
        return f"R$ {float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    @staticmethod
    def gerar_pdf_ficha(ficha: FichaServico) -> bytes:
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import mm
            from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer,
                                            Table, TableStyle)

            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer, pagesize=A4,
                topMargin=12*mm, bottomMargin=14*mm,
                leftMargin=14*mm, rightMargin=14*mm
            )

            styles = getSampleStyleSheet()
            title = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=13, spaceAfter=6, alignment=1)
            small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=9, leading=12)
            tiny  = ParagraphStyle("Tiny",  parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.grey)

            story = []
            story.append(Paragraph("IT0907_01 Formulário de Serviço Técnico de Suporte Viasoft SQV", title))
            story.append(Spacer(1, 4))

            # Cabeçalho compacto
            cabecalho = [
                [Paragraph(f"<b>Ficha:</b> {ficha.id}", small),
                 Paragraph(f"<b>Tipo:</b> {ficha.tipo_servico}", small)],
                [Paragraph(f"<b>Cliente:</b> {ficha.cliente}", small), ""],
            ]
            t_head = Table(cabecalho, colWidths=[95*mm, 75*mm])
            t_head.setStyle(TableStyle([
                ("BOX", (0,0), (-1,-1), 0.5, colors.black),
                ("INNERGRID", (0,0), (-1,-1), 0.5, colors.black),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ]))
            story.append(t_head)
            story.append(Spacer(1, 6))

            # Tabela de registros
            header = ["Data", "Executor", "Início", "Fim", "Total", "Min", "Faturável"]
            col_widths = [22*mm, 52*mm, 20*mm, 20*mm, 22*mm, 14*mm, 20*mm]

            def _table_header():
                t = Table([header], colWidths=col_widths)
                t.setStyle(TableStyle([
                    ("GRID", (0,0), (-1,-1), 0.5, colors.black),
                    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                    ("FONTSIZE", (0,0), (-1,-1), 9),
                    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ]))
                return t

            def _table_row(row):
                t = Table([row], colWidths=col_widths)
                t.setStyle(TableStyle([
                    ("GRID", (0,0), (-1,-1), 0.5, colors.black),
                    ("FONTSIZE", (0,0), (-1,-1), 9),
                    ("ALIGN", (2,0), (-1,-1), "CENTER"),
                    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ]))
                return t

            story.append(_table_header())

            total_cobr_dec = total_ncobr_dec = 0.0
            total_cobr_min = total_ncobr_min = 0

            for r in ficha.registros:
                tot_hms = r.total_hrs or "00:00:00"
                tot_dec = PDFGenerator._hms_to_decimal(tot_hms)
                tot_min = PDFGenerator._hms_to_minutes(tot_hms)

                if r.faturavel:
                    total_cobr_dec += tot_dec
                    total_cobr_min += tot_min
                else:
                    total_ncobr_dec += tot_dec
                    total_ncobr_min += tot_min

                story.append(_table_row([
                    r.data, r.executor, r.hr_inicio, r.hr_fim,
                    tot_hms, str(tot_min), "Sim" if r.faturavel else "Não"
                ]))

                tarefa_txt = ""
                if getattr(r, "tarefa_id", "") or getattr(r, "tarefa_nome", ""):
                    tid = r.tarefa_id or "-"
                    tnm = r.tarefa_nome or ""
                    sep = " — " if (tnm and tid) else ""
                    tarefa_txt = f"<br/><b>Tarefa:</b> #{tid}{sep}{tnm}"

                desc_html = f"<b>Descrição:</b> {r.descricao or ''}{tarefa_txt}"
                desc_tbl = Table([[Paragraph(desc_html, small)]], colWidths=[sum(col_widths)])
                desc_tbl.setStyle(TableStyle([
                    ("BOX", (0,0), (-1,-1), 0.25, colors.grey),
                    ("BACKGROUND", (0,0), (-1,-1), colors.Color(0.97,0.97,0.97)),
                    ("LEFTPADDING", (0,0), (-1,-1), 4),
                    ("RIGHTPADDING", (0,0), (-1,-1), 4),
                    ("TOPPADDING", (0,0), (-1,-1), 3),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                ]))
                story.append(desc_tbl)
                story.append(Spacer(1, 4))

            total_dec = total_cobr_dec + total_ncobr_dec
            total_min = total_cobr_min + total_ncobr_min

            totais_par = Paragraph(
                (
                    f"<b>Horas Cobradas:</b> {PDFGenerator._decimal_to_hms(total_cobr_dec)} "
                    f"({total_cobr_dec:.2f} h | {total_cobr_min} min) &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"<b>Não Cobradas:</b> {PDFGenerator._decimal_to_hms(total_ncobr_dec)} "
                    f"({total_ncobr_dec:.2f} h | {total_ncobr_min} min) &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"<b>Total:</b> {PDFGenerator._decimal_to_hms(total_dec)} "
                    f"({total_dec:.2f} h | {total_min} min)"
                ),
                small
            )

            box_totais = Table([[totais_par]], colWidths=[sum(col_widths)])
            box_totais.setStyle(TableStyle([
                ("BOX", (0,0), (-1,-1), 0.5, colors.black),
                ("BACKGROUND", (0,0), (-1,-1), colors.whitesmoke),
                ("LEFTPADDING", (0,0), (-1,-1), 6),
                ("RIGHTPADDING", (0,0), (-1,-1), 6),
                ("TOPPADDING", (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ]))
            story.append(box_totais)
            story.append(Spacer(1, 6))

            valor_total = (total_cobr_min / 60.0) * float(ficha.valor_hora or 0.0)
            cobranca = [
                [Paragraph(f"<b>Valor/Hora:</b> {PDFGenerator._fmt_money(ficha.valor_hora)}", small)],
                [Paragraph(f"<b>Valor Total Cobrado:</b> {PDFGenerator._fmt_money(valor_total)}", small)],
            ]
            t_cob = Table(cobranca, colWidths=[sum(col_widths)])
            t_cob.setStyle(TableStyle([
                ("BOX", (0,0), (-1,-1), 0.5, colors.black),
                ("LEFTPADDING", (0,0), (-1,-1), 6),
                ("RIGHTPADDING", (0,0), (-1,-1), 6),
                ("TOPPADDING", (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ]))
            story.append(t_cob)
            story.append(Spacer(1, 4))

            story.append(Paragraph("Gerado automaticamente pelo Sistema de Anotações Inteligentes", tiny))

            generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")
            def _footer(canvas, _doc):
                canvas.saveState()
                canvas.setFont("Helvetica", 8)
                canvas.setFillGray(0.35)
                w, h = A4
                left = _doc.leftMargin
                right = w - _doc.rightMargin
                canvas.drawString(left, 10*mm, f"Gerado em {generated_at}")
                canvas.drawRightString(right, 10*mm, f"Página {canvas.getPageNumber()}")
                canvas.restoreState()

            doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
            buffer.seek(0)
            return buffer.getvalue()

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return (f"Erro ao gerar PDF: {e}\n\n{tb}").encode("utf-8")

# =========================
#  Fichas (persistência)
# =========================
class FichaManager:
    """Gerenciador de fichas de serviço"""

    def __init__(self):
        self.fichas_file = "fichas_servico.json"
        self.fichas = self.carregar_fichas()

    def carregar_fichas(self) -> Dict[str, FichaServico]:
        """Carrega fichas do arquivo, ignorando chaves desconhecidas."""
        if not os.path.exists(self.fichas_file):
            return {}

        try:
            with open(self.fichas_file, "r", encoding="utf-8") as f:
                raw = json.load(f)

            fichas: Dict[str, FichaServico] = {}
            for ficha_id, ficha_data in (raw or {}).items():
                # Normaliza registros
                registros_norm = []
                for reg in ficha_data.get("registros", []) or []:
                    reg.setdefault("executor_id", "")
                    reg_norm = _filter_keys(reg, RegistroAtividade)
                    registros_norm.append(RegistroAtividade(**reg_norm))

                ficha_data["registros"] = registros_norm
                # Filtra chaves desconhecidas da ficha
                ficha_norm = _filter_keys(ficha_data, FichaServico)
                fichas[ficha_id] = FichaServico(**ficha_norm)

            return fichas

        except Exception as e:
            st.error(f"Erro ao carregar fichas: {e}")
            return {}

    def salvar_fichas(self):
        try:
            data = {fid: asdict(ficha) for fid, ficha in self.fichas.items()}
            with open(self.fichas_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"Erro ao salvar fichas: {e}")

    def criar_ficha(self, cliente: str, projeto_id: str, vertical: str, tipo_servico: str, valor_hora: float) -> str:
        ficha_id = f"FICHA-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        ficha = FichaServico(
            id=ficha_id,
            cliente=cliente,
            projeto_id=projeto_id,
            vertical=vertical,
            tipo_servico=tipo_servico,
            valor_hora=valor_hora,
            registros=[],
            data_criacao=datetime.now().isoformat(),
            status="Em Andamento",
        )
        self.fichas[ficha_id] = ficha
        self.salvar_fichas()
        return ficha_id

    def adicionar_registro(self, ficha_id: str, registro: RegistroAtividade):
        if ficha_id in self.fichas:
            self.fichas[ficha_id].registros.append(registro)
            self.salvar_fichas()

    def concluir_ficha(self, ficha_id: str):
        if ficha_id in self.fichas:
            self.fichas[ficha_id].status = "Concluída"
            self.salvar_fichas()
            return PDFGenerator.gerar_pdf_ficha(self.fichas[ficha_id])
        return None

# =========================
#  Helpers Teamwork
# =========================
def calcular_total_horas(hr_inicio: str, hr_fim: str) -> str:
    try:
        inicio = datetime.strptime(hr_inicio, "%H:%M")
        fim = datetime.strptime(hr_fim, "%H:%M")
        if fim < inicio:
            fim = fim.replace(day=inicio.day + 1)
        delta = fim - inicio
        total_seg = int(delta.total_seconds())
        h = total_seg // 3600
        m = (total_seg % 3600) // 60
        s = total_seg % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "00:00:00"

def get_project_people_fallback(project_id: str) -> List[Dict[str, str]]:
    base = TEAMWORK_CONFIG['base_url'].rstrip('/')
    url = f"{base}/projects/{project_id}/people.json"
    try:
        resp = requests.get(url, headers=_teamwork_auth_headers(), timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.warning(f"Não foi possível consultar pessoas do projeto: {e}")
        return []

    raw = data.get("people") or data.get("persons") or data.get("users") or []
    people: List[Dict[str, str]] = []
    for p in raw:
        u = p.get("user", p)
        name = (
            u.get("name")
            or f"{u.get('first-name','').strip()} {u.get('last-name','').strip()}".strip()
            or u.get("full-name")
            or u.get("email-address")
            or "Sem nome"
        )
        pid = str(u.get("id") or u.get("userId") or p.get("id") or "")
        if pid and name:
            people.append({"id": pid, "name": name})

    people.sort(key=lambda x: x["name"].lower())
    return people

# ---- Log time no Teamwork (v1)
def _hms_to_h_m(hms: str) -> tuple[int, int]:
    try:
        h, m, s = [int(x) for x in (hms or "0:0:0").split(":")]
    except Exception:
        return 0, 0
    total_min = int(round(h*60 + m + s/60.0))
    return total_min // 60, total_min % 60

def _hms_to_decimal(hms: str) -> str:
    try:
        h, m, s = [int(x) for x in (hms or "0:0:0").split(":")]
    except Exception:
        return "0"
    dec = h + m/60.0 + s/3600.0
    return f"{dec:.2f}"

def _post_time_entry_tw(reg: RegistroAtividade, project_id: str) -> dict:
    base = TEAMWORK_CONFIG['base_url'].rstrip('/')
    headers = {**_teamwork_auth_headers(), "Content-Type": "application/json", "Accept": "application/json"}

    proj_id   = int(project_id) if str(project_id).isdigit() else project_id
    task_id_i = int(reg.tarefa_id)   if str(reg.tarefa_id).isdigit()   else None
    user_id_i = int(reg.executor_id) if str(reg.executor_id).isdigit() else None

    dt = datetime.strptime(reg.data, "%d/%m/%Y")
    date_iso = dt.strftime("%Y-%m-%d")
    date_num = dt.strftime("%Y%m%d")
    time_hm  = (reg.hr_inicio or "00:00")[:5]

    hh, mm   = _hms_to_h_m(reg.total_hrs)
    if hh == 0 and mm == 0:
        mm = 1
    hours_dec = _hms_to_decimal(reg.total_hrs)

    # corta descrição (evita 400 por tamanho)
    descr = (reg.descricao or "").strip()
    if len(descr) > 2000:
        descr = descr[:2000] + "…"

    endpoints = []
    if task_id_i:
        endpoints.append(f"{base}/tasks/{task_id_i}/time_entries.json")
    endpoints.append(f"{base}/projects/{proj_id}/time_entries.json")
    endpoints.append(f"{base}/time_entries.json")

    def _payload_kebab(date_value, use_decimal=False, include_time=True, bill_as_int=False):
        te = {
            "date": date_value,
            "description": descr,
            "isbillable": (1 if reg.faturavel else 0) if bill_as_int else ("1" if reg.faturavel else "0"),
        }
        if include_time:
            te["time"] = time_hm
        if use_decimal:
            te["hours"] = hours_dec
        else:
            te["hours"] = hh
            te["minutes"] = mm
        if task_id_i: te["task-id"]   = task_id_i
        if user_id_i: te["person-id"] = user_id_i
        return {"time-entry": te}

    def _payload_camel(date_value, use_decimal=False, include_time=True):
        te = {
            "date": date_value,
            "description": descr,
            "isBillable": True if reg.faturavel else False,
        }
        if include_time:
            te["time"] = time_hm
        if use_decimal:
            te["hours"] = hours_dec
        else:
            te["hours"] = hh
            te["minutes"] = mm
        if task_id_i: te["taskId"] = task_id_i
        if user_id_i: te["userId"] = user_id_i
        return {"time-entry": te}

    payloads = [
        # v1 "clássico": YYYYMMDD + hours/minutes + isbillable num
        _payload_kebab(date_num, use_decimal=False, include_time=True, bill_as_int=True),
        # variações
        _payload_kebab(date_num, use_decimal=False, include_time=True, bill_as_int=False),
        _payload_kebab(date_iso, use_decimal=False, include_time=True, bill_as_int=True),
        _payload_kebab(date_num, use_decimal=True,  include_time=True, bill_as_int=True),
        _payload_camel(date_num, use_decimal=False, include_time=True),
        _payload_camel(date_num, use_decimal=True,  include_time=True),
        # sem "time"
        _payload_kebab(date_num, use_decimal=False, include_time=False, bill_as_int=True),
        _payload_camel(date_num, use_decimal=False, include_time=False),
    ]

    last_resp = None
    erros = []
    for ep in endpoints:
        for pay in payloads:
            te = {k: v for k, v in pay["time-entry"].items() if v not in ("", None)}
            try:
                r = requests.post(ep, headers=headers, json={"time-entry": te}, timeout=30)
                last_resp = r
                if r.status_code < 400:
                    return r.json()
                erros.append((ep, r.status_code, te, (r.text or "")[:300]))
            except Exception as e:
                erros.append((ep, "EXC", te, str(e)[:300]))

    msg_tail = " | ".join([f"[{st}] {ep} payload={te} resp={txt}" for ep, st, te, txt in erros[-3:]])
    raise requests.HTTPError(f"Todas as variações falharam. Amostras: {msg_tail}", response=last_resp)

PREFS_FILE = "viasell_task_prefs.json"

def _load_task_prefs() -> dict:
    try:
        with open(PREFS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_task_prefs(project_id: str, tasks: list[dict], tag: str, apply_tag: bool):
    prefs = _load_task_prefs()
    prefs[str(project_id)] = {
        "tasks": [{"id": t["id"], "name": t["name"]} for t in tasks],
        "tag": tag,
        "apply_tag": bool(apply_tag),
    }
    with open(PREFS_FILE, "w", encoding="utf-8") as f:
        json.dump(prefs, f, ensure_ascii=False, indent=2)

def _get_prefs_for_project(project_id: str) -> dict | None:
    return _load_task_prefs().get(str(project_id))

def get_all_tasks_project(project_id: str) -> list[dict]:
    """Busca TODAS as tarefas (paginado). Traz 'tags' quando possível."""
    base = TEAMWORK_CONFIG['base_url'].rstrip('/')
    headers = _teamwork_auth_headers()
    all_items = []
    page = 1
    while True:
        # tentativa com tags
        url = f"{base}/projects/{project_id}/tasks.json?pageSize=200&page={page}&includeCompletedTasks=true&include=tags"
        try:
            r = requests.get(url, headers=headers, timeout=25)
            r.raise_for_status()
        except Exception:
            # fallback sem include=tags (evita 400)
            url = f"{base}/projects/{project_id}/tasks.json?pageSize=200&page={page}&includeCompletedTasks=true"
            r = requests.get(url, headers=headers, timeout=25)
            r.raise_for_status()

        data = r.json()
        items = data.get("tasks") or data.get("todo-items") or []
        if not items:
            break
        all_items.extend(items)
        if len(items) < 200:
            break
        page += 1
    return all_items

def _filter_tasks_by_tag(items: list[dict], tag: str) -> list[dict]:
    """Mantém tarefas cujo nome contenha a tag OU tenham a tag em items[].tags[].name."""
    tag_l = (tag or "").strip().lower()
    if not tag_l:
        # normaliza saída (id + name)
        return [{"id": str(i.get("id") or i.get("id_str") or ""), "name": (i.get("content") or i.get("name") or "")} for i in items]
    out = []
    for i in items:
        name = (i.get("content") or i.get("name") or "")
        tid  = str(i.get("id") or i.get("id_str") or "")
        tag_names = [t.get("name","").lower() for t in i.get("tags", [])]
        if tag_l in name.lower() or tag_l in tag_names:
            out.append({"id": tid, "name": name})
    # dedup por id
    seen, dedup = set(), []
    for t in out:
        if t["id"] and t["id"] not in seen:
            seen.add(t["id"]); dedup.append(t)
    return dedup



# =========================
#  NOVA FUNCIONALIDADE: Viasell COM INTERFACE TABULAR CORRIGIDA E BOTÃO FINAL
# =========================
def processar_dados_viasell_tabular(registros_selecionados, tarefas_mapeadas, projeto_id):
    """Processa registros selecionados na interface tabular"""
    sucesso_count = 0
    falha_count = 0
    erros = []

    consultor_id = st.session_state.get(K("consultor_id"), "")
    consultor_nome = st.session_state.get(K("consultor_nome"), "")

    try:
        for reg_index, reg_data in registros_selecionados.items():
            if reg_index in tarefas_mapeadas and tarefas_mapeadas[reg_index]:
                tarefa_id, tarefa_nome = tarefas_mapeadas[reg_index]

                # Criar objeto RegistroAtividade com consultor (se definido)
                registro = RegistroAtividade(
                    id=str(uuid.uuid4()),
                    data=reg_data.get('data', ''),
                    executor=consultor_nome or reg_data.get('executor', ''),
                    executor_id=consultor_id or "",
                    hr_inicio=reg_data.get('hr_inicio', ''),
                    hr_fim=reg_data.get('hr_fim', ''),
                    descricao=reg_data.get('descricao', ''),
                    total_hrs=reg_data.get('total_hrs', ''),
                    faturavel=reg_data.get('faturavel', True),
                    tarefa_id=tarefa_id,
                    tarefa_nome=tarefa_nome
                )

                try:
                    _post_time_entry_tw(registro, projeto_id)
                    sucesso_count += 1
                except Exception as e:
                    falha_count += 1
                    erros.append(f"Registro {reg_index+1} → Tarefa #{tarefa_id}: {str(e)}")

        return sucesso_count, falha_count, erros

    except Exception as e:
        return 0, len(registros_selecionados), [f"Erro geral: {str(e)}"]

def salvar_log_viasell(dados_ficha, sucessos, falhas, erros_detalhados=None, extras: dict | None = None):
    """Salva log da operação de lançamento Viasell (NDJSON)."""
    try:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "cliente": dados_ficha.get("cliente", ""),
            "projeto_id": dados_ficha.get("projeto_id", ""),
            "sucessos": int(sucessos or 0),
            "falhas": int(falhas or 0),
            "total_registros": len(dados_ficha.get("registros", []) or []),
            # enriquecimento de contexto disponível em sessão
            "consultor_nome": st.session_state.get(K("consultor_nome"), ""),
            "consultor_id": st.session_state.get(K("consultor_id"), ""),
        }
        if erros_detalhados:
            entry["erros"] = list(erros_detalhados)

        # extras opcionais (ex.: nome do projeto, nome do arquivo)
        if extras:
            for k, v in extras.items():
                entry[k] = v

        # salva em memória (sessão) e em arquivo
        st.session_state.setdefault("viasell_logs", []).append(entry)
        _append_log(entry, LOG_FILE)

    except Exception as e:
        st.error(f"Erro ao salvar log: {e}")


def extrair_numero_ficha(texto: str) -> str:
    """
    Extrai o número da Ficha do texto (ex.: 'Ficha: 12345', 'FICHA - 000987').
    Retorna string vazia se não encontrar.
    """
    padroes = [
        r'\bFicha\s*[:\-#]*\s*(\d{3,})',
        r'\bFICHA\s*[:\-#]*\s*(\d{3,})',
        r'\bN[ºo]\s*[:\-#]*\s*(\d{3,})\s*(?:Ficha|FICHA)?',
    ]
    for p in padroes:
        m = re.search(p, texto, re.IGNORECASE)
        if m:
            num = m.group(1)
            # remove zeros à esquerda, mas mantém '0' se tudo zero
            return num.lstrip('0') or num
    return ""


def buscar_descricao_serv_exec(linhas: List[str], linha_registro_idx: int) -> str:
    """
    Captura a descrição da seção 'Serviço Exec.' associada ao registro encontrado
    na linha `linha_registro_idx`. Lê a linha do cabeçalho e continua coletando o
    texto das linhas seguintes até encontrar um delimitador (novo registro, totais etc.).
    """
    inicio_busca = max(0, linha_registro_idx - 2)
    fim_busca = min(len(linhas), linha_registro_idx + 25)

    def _eh_inicio_novo_registro(l: str) -> bool:
        return bool(re.match(r'\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\s+\d+\s*-\s+', l))

    def _eh_delimitador(l: str) -> bool:
        return bool(re.search(
            r'(Hr\(\s*\)\s*Tec|Total\s+Hr|Valor\s*R\$|Informações\s+de\s+Cobrança|Assinatura|Formulário|Ficha|Cliente\s*\d{3,})',
            l, re.IGNORECASE
        ))

    desc_partes: List[str] = []

    for i in range(inicio_busca, fim_busca):
        linha = (linhas[i] or "").strip()

        # Cabeçalho "Serviço Exec"
        if re.search(r'Serv[ií]?[çc]o\s+Exec', linha, re.IGNORECASE):
            # Tenta pegar o conteúdo na mesma linha após ":" / "-" etc.
            m = re.search(r'Serv[ií]?[çc]o\s+Exec[^:]*[:\-\–]\s*(.*)$', linha, re.IGNORECASE)
            if m and m.group(1).strip():
                desc_partes.append(m.group(1).strip())

            # Coleta as próximas linhas até um delimitador
            for j in range(i + 1, min(i + 30, len(linhas))):
                l2 = (linhas[j] or "").strip()
                if not l2:
                    continue
                if _eh_inicio_novo_registro(l2) or _eh_delimitador(l2):
                    break
                if re.match(r'^[\-\=_.\s]{3,}$', l2):  # separadores
                    continue
                if l2.isdigit():  # linhas só com número
                    continue
                desc_partes.append(l2)
            break

    if not desc_partes:
        return ""

    texto = " ".join(desc_partes)
    texto = re.sub(r'\s+', ' ', texto).strip()
    texto = re.sub(r'[,\s\-_]+$', '.', texto)
    if texto and not texto.endswith(('.', '!', '?', ':')):
        texto += '.'
    return texto

def _reset_nova_ficha():
    # limpa todo estado do namespace nvf:
    for k in [k for k in list(st.session_state.keys()) if k.startswith(NVF_NS)]:
        st.session_state.pop(k, None)

    # remove qualquer key do file_uploader atual (com ou sem nonce)
    up_prefix = f"{K('viasell_upload')}::"
    for k in list(st.session_state.keys()):
        if k.startswith(up_prefix) or k == K("viasell_upload"):
            st.session_state.pop(k, None)

    # limpa flags soltas
    st.session_state.pop("confirma_lancamento", None)
    st.session_state.pop(K("confirm_clear_hist"), None)
    st.session_state.pop(K("mostrar_picker_consultor"), None)

    # força remontagem do file_uploader trocando o nonce
    st.session_state[K("uploader_nonce")] = st.session_state.get(K("uploader_nonce"), 0) + 1

    # opcional: se usar @st.cache_data em algum lugar, descomente:
    # st.cache_data.clear()

def lancar_fichas_viasell():
    """Página para lançamento de fichas Viasell com interface tabular corrigida"""
    st.header("🏭 Lançar Fichas")
    st.markdown("**Ferramenta para upload e processamento de fichas do Viasell para Teamwork**")
    
    # Verificar se configurações estão definidas
    config_manager = st.session_state.get("config_manager")
    if not config_manager:
        st.error("❌ Configurações não carregadas. Acesse a aba Configurações primeiro.")
        return
    
    config = config_manager.config
    
    # Verificar dependências
    if not HAS_PDF:
        st.warning("⚠️ Para processar PDFs, instale pdfplumber: `pip install pdfplumber`")
    
    # ===== UPLOAD DE ARQUIVO =====
    st.subheader("1 - 📁 Upload da Ficha")

    nonce = st.session_state.get(K("uploader_nonce"), 0)
    
    uploaded_file = st.file_uploader(
        "Selecione a ficha do Viasell:",
        type=['pdf', 'json', 'txt', 'csv'],
        key=f"{K('viasell_upload')}::{nonce}"  # <— força remontagem quando o nonce muda
    )
    
    if not uploaded_file:
        st.info("👆 Faça upload de uma ficha gerada pelo Viasell para continuar")
        return
    
    # ===== PROCESSAMENTO DO ARQUIVO =====
    #st.subheader("🔍 Processamento da Ficha")
    
    try:
        with st.spinner("Extraindo dados do arquivo..."):
            # Ler conteúdo do arquivo
            file_content = extract_text_from_upload(uploaded_file)
            
            # Debug do texto extraído
            debug_texto_extraido(file_content)
            
            if uploaded_file.type == "application/json":
                try:
                    dados_extraidos = json.loads(file_content)
                except json.JSONDecodeError:
                    st.error("Arquivo JSON inválido")
                    return
            else:
                # Extrair dados do texto com função corrigida
                if getattr(config, "ocultar_painel_analise", True):
                    with _mute_debug_ui():
                        dados_extraidos = extrair_dados_viasell_corrigido(file_content)
                else:
                    dados_extraidos = extrair_dados_viasell_corrigido(file_content)
                # Aplicar configurações padrão se necessário
                if not dados_extraidos.get('vertical'):
                    dados_extraidos['vertical'] = config.vertical_padrao
                if not dados_extraidos.get('valor_hora'):
                    dados_extraidos['valor_hora'] = config.valor_hora_padrao
        
        st.success("✅ Arquivo processado com sucesso!")
        
        # ===== SELEÇÃO DE PROJETO (SIMPLIFICADA) =====
        st.subheader("2 - 🎯 Seleção de Projeto e Consultor")
        
        # Buscar projetos do Teamwork usando API v1
        with st.spinner("Buscando projetos no Teamwork..."):
            projetos = get_projects_teamwork()
        
        if projetos:
            # Criar dropdown com projetos
            projeto_opcoes = {}
            for proj in projetos:
                categoria_info = f" ({proj['category']})" if proj['category'] else ""
                label = f"{proj['name']}{categoria_info} - #{proj['id']}"
                projeto_opcoes[label] = proj
            
            projeto_selecionado = st.selectbox(
                "📂 Selecione o projeto:",
                options=[""] + list(projeto_opcoes.keys()),
                key=K("projeto_select"),
                help="Projetos carregados do Teamwork via API v1"
            )
            
            projeto_info = projeto_opcoes.get(projeto_selecionado, {}) if projeto_selecionado else {}
            projeto_id = projeto_info.get('id', '')
        else:
            st.error("❌ Não foi possível carregar projetos do Teamwork")
            projeto_id = ""
            projeto_info = {}
# ===== CONSULTOR (logo após selecionar o projeto) =====
        consultor_nome_atual = st.session_state.get(K("consultor_nome"))
        consultor_id_atual   = st.session_state.get(K("consultor_id"))
        obrigatorio = getattr(config, "exigir_consultor", True)
        
        if projeto_id:
            "👤 Selecione o Consultor responsável"


            colc1, colc2 = st.columns([1, 2])
            with colc1:
                if st.button("👤 Seleciona consultor responsável", key=K("btn_definir_consultor")):
                    st.session_state[K("mostrar_picker_consultor")] = True


            with colc2:
                if consultor_nome_atual:
                    st.info(
                        f"Consultor atual: {consultor_nome_atual}"
                        + (f" (#{consultor_id_atual})" if consultor_id_atual else "")
                    )


        if st.session_state.get(K("mostrar_picker_consultor"), False):
                with st.spinner("Carregando pessoas do projeto..."):
                    pessoas = get_project_people_fallback(projeto_id)

                if pessoas:
                    opcoes = {f"{p['name']} (#{p['id']})": p for p in pessoas}
                    escolhido = st.selectbox(
                        "Selecione o consultor do projeto:",
                        options=[""] + list(opcoes.keys()),
                        key=K("consultor_select")
                    )
                    if escolhido:
                        p = opcoes[escolhido]
                        st.session_state[K("consultor_id")] = p["id"]
                        st.session_state[K("consultor_nome")] = p["name"]
                        st.session_state[K("mostrar_picker_consultor")] = False
                        st.success(f"Consultor definido: {p['name']} (#{p['id']})")
                        st.rerun()
                else:
                    st.warning("Não encontrei pessoas no projeto. Informe manualmente:")
                    nome_manual = st.text_input("Nome do consultor:", key=K("consultor_manual"))
                    if nome_manual:
                        st.session_state[K("consultor_id")] = ""
                        st.session_state[K("consultor_nome")] = nome_manual
                        st.session_state[K("mostrar_picker_consultor")] = False
                        st.success(f"Consultor definido: {nome_manual}")
                        st.rerun()
        
                    if obrigatorio and not (consultor_nome_atual or st.session_state.get(K("consultor_manual"))):
                        st.warning("⚠️ Consultor obrigatório. Clique em **Seleciona consultor responsável** ou informe manualmente.")
        # ===== VISUALIZAÇÃO DOS DADOS =====
        if dados_extraidos and projeto_id:
            st.markdown("**📊 Dados Extraídos da Ficha**")
            
            # Mostrar dados básicos
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Cliente:** {dados_extraidos.get('cliente', 'N/A')}")
                st.write(f"**Projeto Selecionado:** {projeto_info.get('name', 'N/A')} (#{projeto_id})")
                st.write(f"**Consultor:** {st.session_state.get(K('consultor_nome'), '—')}")
                st.write(f"**Vertical:** {dados_extraidos.get('vertical', 'N/A')}")
                
            with col2:
                st.write(f"**Tipo Serviço:** {dados_extraidos.get('tipo_servico', 'N/A')}")
                st.write(f"**Valor/Hora:** R$ {dados_extraidos.get('valor_hora', 0.0):.2f}")
                st.write(f"**Total Registros:** {len(dados_extraidos.get('registros', []))}")
            
            # ===== BUSCAR TAREFAS NO TEAMWORK =====
            registros = dados_extraidos.get('registros', [])
            if registros:
                # Buscar tarefas usando a tag configurada
                with st.spinner(f"Buscando tarefas com tag '{config.tag_teamwork}' no projeto {projeto_id}..."):
                    tarefas = get_tasks_by_tag_and_project(projeto_id, config.tag_teamwork,inherit_from_parent=True)
                
                if tarefas:
                    st.success(f"✅ Encontradas {len(tarefas)} tarefas com a tag '{config.tag_teamwork}'")
                    
                    # ===== INTERFACE TABULAR =====
                    st.subheader("3 - 📋 Lançamentos")
                    st.markdown("**Selecione os registros a lançar e mapeie as tarefas correspondentes da EAP:**")
                    
                    # Preparar opções de tarefas para dropdown
                    tarefa_opcoes = {f"{t['name']} (#{t['id']})": (t['id'], t['name']) for t in tarefas}
                    opcoes_lista = ["Selecione uma tarefa..."] + list(tarefa_opcoes.keys())
                    
                    # CSS customizado para tabela
                    st.markdown("""
                    <style>
                    .registro-row {
                        border: 1px solid #e0e0e0;
                        border-radius: 20px;
                        padding: 10px;
                        margin: 5px 0;
                        background-color: #f9f9f9;
                    }
                    .registro-header {
                        font-weight: bold;
                        color: #2e3a46;
                        background-color: #e8f4fd;
                        padding: 10px;
                        border-radius: 20px;
                        margin: 5px 0;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    registros_selecionados = {}
                    tarefas_mapeadas = {}
                    
                    # Cabeçalho da tabela
                    st.markdown('<div class="registro-header">', unsafe_allow_html=True)
                    col_head = st.columns([1, 1.5, 1, 1, 2.5, 1, 2])
                    with col_head[0]:
                        st.markdown("**Selecionar**")
                    with col_head[1]:
                        st.markdown("**Data / Executor**")
                    with col_head[2]:
                        st.markdown("**Início**")
                    with col_head[3]:
                        st.markdown("**Fim**")
                    with col_head[4]:
                        st.markdown("**Descrição**")
                    with col_head[5]:
                        st.markdown("**Faturável**")
                    with col_head[6]:
                        st.markdown("**Tarefa EAP (Tag)**")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Renderizar cada registro
                    for i, reg in enumerate(registros):
                        st.markdown(f'<div class="registro-row">', unsafe_allow_html=True)
                        
                        # Linha principal com checkbox e dados
                        col_check, col_data, col_inicio, col_fim, col_desc, col_fat, col_tarefa = st.columns([1, 1.5, 1, 1, 2.5, 1, 2])
                        
                        with col_check:
                            selecionado = st.checkbox(
                                "",
                                value=st.session_state.get(K(f"checkbox_{i}"), False),
                                key=K(f"checkbox_{i}"),
                                label_visibility="collapsed"
                            )
                        
                        with col_data:
                            st.markdown(f"**{reg.get('data', 'N/A')}**")
                            st.caption(reg.get('executor', 'N/A'))
                        
                        with col_inicio:
                            st.write(reg.get('hr_inicio', 'N/A'))
                        
                        with col_fim:
                            st.write(reg.get('hr_fim', 'N/A'))
                        
                        with col_desc:
                            descricao = reg.get('descricao', '')
                            if len(descricao) > 60:
                                descricao_preview = descricao[:60] + "..."
                                st.write(descricao_preview)
                                with st.expander("Ver mais"):
                                    st.write(descricao)
                            else:
                                st.write(descricao)
                            st.caption(f"Total: {reg.get('total_hrs', 'N/A')}")
                        
                        with col_fat:
                            st.write("✅ Sim" if reg.get('faturavel', True) else "❌ Não")
                        
                        with col_tarefa:
                            tarefa_selecionada = st.selectbox(
                                "",
                                options=opcoes_lista,
                                key=K(f"tarefa_{i}"),
                                label_visibility="collapsed"
                            )
                            
                            if tarefa_selecionada != "Selecione uma tarefa...":
                                tarefa_id, tarefa_nome = tarefa_opcoes[tarefa_selecionada]
                                st.caption(f"#{tarefa_id}")
                        
                        # Armazenar seleções
                        if selecionado:
                            registros_selecionados[i] = reg
                            if tarefa_selecionada != "Selecione uma tarefa...":
                                tarefas_mapeadas[i] = tarefa_opcoes[tarefa_selecionada]
                        
                        st.markdown('</div>', unsafe_allow_html=True)

                    # ===== BOTÃO FINAL - LANÇAR FICHA COMPLETA =====
                    st.divider()
                    st.subheader("4 - 🏁 Finalizar Ficha Viasell")
                    
                    # Verificar se há registros válidos para lançamento
                    registros_com_tarefa_final = {k: v for k, v in tarefas_mapeadas.items() if k in registros_selecionados}
                    consultor_nome = st.session_state.get(K("consultor_nome")) or st.session_state.get(K("consultor_manual")) or ""
                    bloqueado_por_consultor = getattr(config, "exigir_consultor", False) and not consultor_nome
                    
                    if registros_com_tarefa_final:
                        # Calcular totais da ficha
                        total_horas_str = "00:00:00"
                        total_valor = 0.0
                        registros_faturáveis = 0
                        
                        for idx in registros_com_tarefa_final.keys():
                            reg = registros[idx]
                            if reg.get('faturavel', True):
                                registros_faturáveis += 1
                                # Somar horas (convertendo HH:MM:SS para minutos)
                                total_hrs = reg.get('total_hrs', '00:00:00')
                                try:
                                    h, m, s = [int(x) for x in total_hrs.split(':')]
                                    minutos_reg = h * 60 + m + s // 60
                                    valor_reg = (minutos_reg / 60.0) * dados_extraidos.get('valor_hora', 180.0)
                                    total_valor += valor_reg
                                except:
                                    pass
                        
                        # Mostrar resumo final da ficha
                        col_resumo, col_botao_final = st.columns([2, 1])
                        
                        with col_resumo:
                            st.markdown("""
                            📊 Resumo Final da Ficha
                            """)
                            
                            col_dados1, col_dados2 = st.columns(2)
                            
                            with col_dados1:
                                st.markdown(f"""
                                **📋 Dados da Ficha:**
                                - **Cliente:** {dados_extraidos.get('cliente', 'N/A')}
                                - **Projeto:** {projeto_info.get('name', 'N/A')} (#{projeto_id})
                                - **Tipo:** {dados_extraidos.get('tipo_servico', 'N/A')}
                                - **Vertical:** {dados_extraidos.get('vertical', 'N/A')}
                                """)
                            
                            with col_dados2:
                                st.markdown(f"""
                                **💰 Resumo Financeiro:**
                                - **Valor/Hora:** R$ {dados_extraidos.get('valor_hora', 180.0):.2f}
                                - **Registros Faturáveis:** {registros_faturáveis}
                                - **Valor Total Estimado:** R$ {total_valor:.2f}
                                - **Registros p/ Lançar:** {len(registros_com_tarefa_final)}
                                """)
                            
                            # Lista detalhada dos registros
                            with st.expander("📋 Detalhes dos Registros para Lançamento"):
                                for idx, (tarefa_id, tarefa_nome) in registros_com_tarefa_final.items():
                                    reg = registros[idx]
                                    faturavel_icon = "💰" if reg.get('faturavel', True) else "🆓"
                                    st.write(f"{faturavel_icon} **{reg.get('data', '')}** - {reg.get('executor', '')} ({reg.get('total_hrs', '')})")
                                    st.write(f"   └ 🎯 **Tarefa:** #{tarefa_id} - {tarefa_nome}")
                                    st.write(f"   └ 📝 **Descrição:** {reg.get('descricao', '')[:100]}...")
                                    st.write("")
                            
                            bloqueado_por_consultor = getattr(config, "exigir_consultor", False) and not consultor_nome
                            if bloqueado_por_consultor:
                                st.warning("⚠️ Defina o consultor para habilitar o lançamento.")
                            
                            # Botão principal de lançamento final
                            if st.button(
                                    "🚀 **LANÇAR FICHA**\n\n**NO TEAMWORK**",
                                    type="primary",
                                    key=K("launch_ficha_completa"),
                                    help=f"Lançar {len(registros_com_tarefa_final)} registros no Teamwork",
                                    use_container_width=True,
                                    disabled=bloqueado_por_consultor
                            ):



                                    # Realizar lançamento completo
                                    st.session_state.confirma_lancamento = False
                                    
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    with st.spinner("🚀 Lançando ficha completa no Teamwork..."):
                                        sucesso_total = 0
                                        falhas_total = 0
                                        erros_detalhados = []
                                        consultor_id = st.session_state.get(K("consultor_id"), "")
                                        consultor_nome = st.session_state.get(K("consultor_nome"), "")
                                        # Lançar registro por registro com progress
                                        for i, (reg_index, reg_data) in enumerate(registros_selecionados.items()):
                                            if reg_index in tarefas_mapeadas and tarefas_mapeadas[reg_index]:
                                                # Atualizar progress
                                                progress = (i + 1) / len(registros_selecionados)
                                                progress_bar.progress(progress)
                                                status_text.text(f"Lançando registro {i+1}/{len(registros_selecionados)}: {reg_data.get('executor', '')} - {reg_data.get('data', '')}")
                                                
                                                tarefa_id, tarefa_nome = tarefas_mapeadas[reg_index]
                                                
                                                # Criar objeto RegistroAtividade
                                                registro = RegistroAtividade(
                                                    id=str(uuid.uuid4()),
                                                    data=reg_data.get('data', ''),
                                                    executor=consultor_nome or reg_data.get('executor', ''),
                                                    executor_id=consultor_id or "",
                                                    hr_inicio=reg_data.get('hr_inicio', ''),
                                                    hr_fim=reg_data.get('hr_fim', ''),
                                                    descricao=reg_data.get('descricao', ''),
                                                    total_hrs=reg_data.get('total_hrs', ''),
                                                    faturavel=reg_data.get('faturavel', True),
                                                    tarefa_id=tarefa_id,
                                                    tarefa_nome=tarefa_nome
                                                )
                                                
                                                try:
                                                    # Lançar no Teamwork
                                                    _post_time_entry_tw(registro, projeto_id)
                                                    sucesso_total += 1
                                                    
                                                except Exception as e:
                                                    falhas_total += 1
                                                    erro_msg = f"Registro {reg_index+1} ({reg_data.get('executor', '')}) → Tarefa #{tarefa_id}: {str(e)}"
                                                    erros_detalhados.append(erro_msg)
                                    
                                    # Limpar progress
                                    progress_bar.empty()
                                    status_text.empty()
                                    
                                    # Mostrar resultado final
                                    if sucesso_total > 0:
                                        st.success(f"🎉 **FICHA LANÇADA COM SUCESSO!**\n\n✅ {sucesso_total} registros lançados no Teamwork!")
                                        
                                        
                                        # Criar resumo de conclusão
                                        st.markdown(f"""
                                        ### 📋 Ficha Processada com Sucesso!
                                        
                                        **Cliente:** {dados_extraidos.get('cliente', 'N/A')}  
                                        **Projeto:** {projeto_info.get('name', 'N/A')} (#{projeto_id})  
                                        **Data/Hora:** {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}  
                                        **Registros Lançados:** {sucesso_total}  
                                        **Valor Total:** R$ {total_valor:.2f}  
                                        **Consultor:** {st.session_state.get(K('consultor_nome'), '—')}
                                        """)
                                    
                                    if falhas_total > 0:
                                        st.error(f"⚠️ {falhas_total} registros falharam no lançamento")
                                        
                                        with st.expander("🔍 Ver detalhes dos erros"):
                                            for erro in erros_detalhados:
                                                st.write(f"❌ {erro}")
                                    
                                    # Salvar log final
                                    salvar_log_viasell(
                                        dados_extraidos,
                                        sucesso_total,
                                        falhas_total,
                                        erros_detalhados,
                                        extras={
                                            "projeto_nome": projeto_info.get("name", ""),
                                            "arquivo": getattr(uploaded_file, "name", ""),
                                        },
                                    )
                                    
                                    # Mostrar estatísticas finais
                                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                                    
                                    with col_stats1:
                                        st.metric("✅ Sucessos", sucesso_total)
                                    with col_stats2:
                                        st.metric("❌ Falhas", falhas_total)
                                    with col_stats3:
                                        taxa_sucesso = (sucesso_total / (sucesso_total + falhas_total) * 100) if (sucesso_total + falhas_total) > 0 else 0
                                        st.metric("📊 Taxa de Sucesso", f"{taxa_sucesso:.1f}%")
                                    
                                    # Opção de novo lançamento
                                    if st.button("🔄 Processar Nova Ficha", key=K("nova_ficha_btn"),type="secondary",use_container_width=True,on_click=_reset_nova_ficha,
                                                 ):
                                        # Limpar state e rerun
                                        for k in [k for k in list(st.session_state.keys()) if k.startswith(NVF_NS)]:
                                            del st.session_state[k]
                                        
                                        for k in list(st.session_state.keys()):
                                            if k.startswith(f"{K('viasell_upload')}::") or k == K("viasell_upload"):
                                                st.session_state.pop(k, None)
                                        
                                                                                
                                        # Zera flags soltas (fora do namespace), se existirem
                                        st.session_state.pop("confirma_lancamento", None)
                                        st.session_state.pop(K("confirm_clear_hist"), None)
                                        
                                        # Força remontagem do file_uploader
                                        st.session_state[K("uploader_nonce")] = st.session_state.get(K("uploader_nonce"), 0) + 1
                                        
                                        st.rerun()
                                        
                            # Botão de cancelar se estiver em modo confirmação
                            if st.session_state.get('confirma_lancamento', False):
                                if st.button("❌ Cancelar", key=K("cancelar_lancamento")):
                                    st.session_state.confirma_lancamento = False
                                    st.rerun()
                    
                    else:
                        # Caso não tenha registros válidos
                        st.warning("⚠️ **Nenhum registro válido para lançamento")
                        st.info("""
                        **Para finalizar a ficha:**
                        1. ✅ Selecione os registros desejados
                        2. 🎯 Mapeie cada registro para uma tarefa
                        3. 🚀 Use o botão 'Lançar Ficha Completa'
                        """)
                
                else:
                    st.warning(f"⚠️ Nenhuma tarefa encontrada com a tag '{config.tag_teamwork}' no projeto selecionado")
                    st.info("💡 Opções:\n- Verifique se a tag existe nas tarefas do projeto\n- Tente buscar sem tag (deixar em branco nas configurações)\n- Verifique se você tem permissão para acessar o projeto")
                    
                    # Opção de buscar todas as tarefas
                    if st.button("🔍 Buscar TODAS as tarefas do projeto", key=K("buscar_todas_tarefas")):
                        with st.spinner("Buscando todas as tarefas..."):
                            todas_tarefas = get_tasks_by_tag_and_project(projeto_id, "")  # Sem filtro de tag
                        if todas_tarefas:
                            st.success(f"✅ Encontradas {len(todas_tarefas)} tarefas no total")
                            st.info("💡 Configure uma tag específica nas Configurações para filtrar as tarefas relevantes")
                        else:
                            st.error("❌ Nenhuma tarefa encontrada no projeto")
                            
            else:
                st.warning("⚠️ Nenhum registro encontrado na ficha")
                st.info("💡 Possíveis causas:\n- Formato da ficha diferente do esperado\n- Problemas na extração de texto do PDF\n- Dados não estão no formato tabular esperado")
                
        elif dados_extraidos and not projeto_id:
            st.info("👆 Selecione um projeto para continuar")
        else:
            st.error("❌ Não foi possível extrair dados da ficha")
            
    except Exception as e:
        st.error(f"❌ Erro ao processar arquivo: {str(e)}")
        with st.expander("Detalhes do erro"):
            import traceback
            st.code(traceback.format_exc())

def historico_lancamentos():
    """Página de histórico dos lançamentos (arquivo JSON/NDJSON)."""
    st.header("📚 Histórico")

    # carrega do arquivo sempre que abrir a página (e guarda em cache de sessão)
    logs = _read_logs(LOG_FILE)
    st.session_state["viasell_logs"] = logs

    if not logs:
        st.info("Nenhum lançamento registrado ainda.")
        return

    # parsing básico
    for x in logs:
        ts = x.get("timestamp")
        try:
            x["_dt"] = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
        except Exception:
            x["_dt"] = None

    # filtros
    colf1, colf2, colf3 = st.columns([1.2, 1, 1])
    with colf1:
        termo = st.text_input("Buscar (cliente / projeto / consultor / arquivo):", "")
    with colf2:
        # datas
        datas_validas = [z["_dt"].date() for z in logs if z.get("_dt")]
        if datas_validas:
            d_min, d_max = min(datas_validas), max(datas_validas)
        else:
            d_min = d_max = datetime.now().date()
        periodo = st.date_input("Período", value=(d_min, d_max))
        if isinstance(periodo, tuple):
            d_inicio, d_fim = periodo
        else:
            d_inicio, d_fim = d_min, periodo
    with colf3:
        nomes = sorted({(z.get("consultor_nome") or "").strip() for z in logs if z.get("consultor_nome")})
        filtro_cons = st.multiselect("Consultor", nomes, default=[])

    # aplica filtros
    def _match(lg: dict) -> bool:
        ok_dt = True
        if lg.get("_dt"):
            d = lg["_dt"].date()
            ok_dt = (d >= d_inicio and d <= d_fim)
        txt = f"{lg.get('cliente','')} {lg.get('projeto_id','')} {lg.get('projeto_nome','')} {lg.get('consultor_nome','')} {lg.get('arquivo','')}".lower()
        ok_txt = (termo.lower() in txt) if termo else True
        ok_cons = (not filtro_cons) or ((lg.get("consultor_nome") or "").strip() in set(filtro_cons))
        return ok_dt and ok_txt and ok_cons

    filtrados = [l for l in logs if _match(l)]
    filtrados.sort(key=lambda z: (z.get("_dt") or datetime.min), reverse=True)

    # métricas
    tot_sessions = len(filtrados)
    tot_ok = sum(int(z.get("sucessos") or 0) for z in filtrados)
    tot_fail = sum(int(z.get("falhas") or 0) for z in filtrados)
    taxa = (tot_ok / (tot_ok + tot_fail) * 100) if (tot_ok + tot_fail) else 0.0

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Sessões de lançamento", tot_sessions)
    with c2: st.metric("Registros OK", tot_ok)
    with c3: st.metric("Taxa de sucesso", f"{taxa:.1f}%")

    st.divider()

    # tabela
    cols_show = ["timestamp","cliente","projeto_nome","projeto_id","consultor_nome","arquivo","sucessos","falhas","total_registros"]
    if HAS_PANDAS:
        import pandas as _pd
        df = _pd.DataFrame([{k: v for k, v in x.items() if k in cols_show} for x in filtrados])
        if not df.empty:
            df = df.sort_values("timestamp", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
        # downloads
        json_txt = _export_logs_json_array([{k: v for k, v in x.items() if k in cols_show or k == "erros"} for x in filtrados])
        st.download_button("⬇️ Baixar JSON", data=json_txt, file_name="historico_viasell.json", mime="application/json")
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Baixar CSV", data=csv_bytes, file_name="historico_viasell.csv", mime="text/csv")
    else:
        st.write("Resultados:")
        for lg in filtrados:
            with st.expander(f"🗓 {lg.get('timestamp','')} • {lg.get('cliente','')} • Proj #{lg.get('projeto_id','')} • {lg.get('consultor_nome','') or '—'}"):
                st.write({k: lg.get(k) for k in cols_show})
                if lg.get("erros"):
                    st.write("Erros:", lg["erros"])

    # ferramentas
    with st.expander("Ferramentas do histórico"):
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Recarregar", use_container_width=True):
                st.rerun()
        with col_b:
            if st.button("🧹 Limpar histórico (confirmação dupla)", use_container_width=True, key=K("btn_clear_hist")):
                st.session_state[K("confirm_clear_hist")] = True
            if st.session_state.get(K("confirm_clear_hist")):
                if st.button("❗ Confirmar limpeza"):
                    try:
                        open(LOG_FILE, "w", encoding="utf-8").close()
                        st.session_state["viasell_logs"] = []
                        st.session_state[K("confirm_clear_hist")] = False
                        st.success("Histórico limpo.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Falha ao limpar: {e}")


# =========================
#  App Principal
# =========================
def main():
    st.set_page_config(
        page_title="Sistema de Anotações Inteligentes",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🤖 Sistema de Anotações Inteligentes")

    # Inicializar componentes e estado
    if "ml_analyzer" not in st.session_state:
        st.session_state.ml_analyzer = MLAnalyzer()
    if "teamwork_client" not in st.session_state:
        st.session_state.teamwork_client = TeamworkClient(
            base_url=TEAMWORK_CONFIG["base_url"],
            api_key=TEAMWORK_CONFIG["api_key"]
        )
    if "ficha_manager" not in st.session_state:
        st.session_state.ficha_manager = FichaManager()
    if "config_manager" not in st.session_state:
        st.session_state.config_manager = ConfigManager()
    if "ficha_atual" not in st.session_state:
        st.session_state.ficha_atual = None
    if "registros_temp" not in st.session_state:
        st.session_state.registros_temp = []

    # Sidebar
    with st.sidebar:
        st.title("🌐 Navegação")
        opcao = st.selectbox(
            "Escolha uma opção:",
            [
                "📋 Integrar Fichas",
                "💾 Histórico", 
                "⚙️ Configurações"
            ],
            key=K("menu")
        )

    # Roteamento
    if opcao == "📋 Integrar Fichas":
        lancar_fichas_viasell()
    elif opcao == "Nova Ficha de Serviço":
        st.info("Funcionalidade de Nova Ficha de Serviço disponível - implementação mantida do código anterior")
    elif opcao == "Fichas em Andamento":
        st.info("Funcionalidade de Fichas em Andamento disponível - implementação mantida do código anterior")
    elif opcao == "💾 Histórico":
        historico_lancamentos()
    elif opcao == "⚙️ Configurações":
        configuracoes_sistema()

def configuracoes_sistema():
    """Página de configurações do sistema"""
    st.header("⚙️ Configurações do Sistema")

    config_manager = st.session_state.config_manager
    config = config_manager.config

    # Garante campo no config (evita KeyError/AttributeError)
    if not hasattr(config, "tarefas_por_projeto") or not isinstance(getattr(config, "tarefas_por_projeto"), dict):
        config.tarefas_por_projeto = {}

    # ===== CONFIGURAÇÕES GERAIS =====
    st.markdown("Configure os valores padrão para integração")

    with st.form(K("config_viasell_form")):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Configurações Teamwork:**")
            nova_tag = st.text_input(
                "Tag padrão para buscar tarefas na EAP:",
                value=getattr(config, "tag_teamwork", ""),
                help="Tag que será usada para filtrar tarefas no Teamwork (deixe em branco para buscar todas)"
            )

        with col2:
            st.write("**Validações de fluxo:**")
            exigir_consultor_novo = st.checkbox(
                "Exigir consultor para lançar fichas",
                value=getattr(config, "exigir_consultor", False),
                help="Quando ativo, o botão 'LANÇAR FICHA COMPLETA' só habilita se um consultor estiver definido."
            )
            ocultar_painel_analise_novo = st.checkbox(
                "Ocultar painel “Analisando texto extraído”",
                value=getattr(config, "ocultar_painel_analise", True),
                key=K("ocultar_painel_analise")
            )

        submitted = st.form_submit_button("💾 Salvar Configurações", type="primary", use_container_width=True)

    if submitted:
        config.tag_teamwork = nova_tag
        config.exigir_consultor = bool(exigir_consultor_novo)
        config.ocultar_painel_analise = bool(ocultar_painel_analise_novo)
        config_manager.salvar_config()
        st.success("✅ Configurações salvas com sucesso!")
        st.rerun()

    # ===== PRÉ-SELEÇÃO DE TAREFAS POR PROJETO (Whitelist) =====
    st.subheader("🗂 Pré-seleção de tarefas por projeto")

    with st.expander("Configurar tarefas que podem receber apontamento (por projeto)", expanded=False):
        # 1) Projeto
        with st.spinner("Carregando projetos..."):
            projetos_cfg = get_projects_teamwork()

        mapa_proj = {f"{p['name']} (#{p['id']})": p["id"] for p in (projetos_cfg or [])}
        label_proj_sel = st.selectbox(
            "Projeto",
            options=[""] + list(mapa_proj.keys()),
            key=K("whitelist_proj_select")
        )
        proj_id_sel = mapa_proj.get(label_proj_sel, "")

        if proj_id_sel:
            st.caption("Dica: esta busca é feita só aqui nas Configurações; no fluxo principal o app usa o que ficar salvo 👍")

            # 2) Filtros e Ação
            colf = st.columns([1, 1, 2])
            with colf[1]:
                filtro_tag_cfg = st.text_input(
                    "Filtro por tag (opcional)",
                    value=getattr(config, "tag_teamwork", ""),
                    key=K("whitelist_tag_filter")
                )
            with colf[2]:
                filtro_txt = st.text_input("Filtro por nome (opcional)", key=K("whitelist_filter_name"))
            with colf[0]:
                do_fetch = st.button("🔄 Carregar/Atualizar tarefas", key=K("whitelist_fetch_btn"))

            # 3) Cache por projeto + tag (evita refetch a cada rerun)
            cache_key = K(f"whitelist_cache::{proj_id_sel}::{(filtro_tag_cfg or '').strip().lower()}")
            tarefas_opcoes = st.session_state.get(cache_key, [])

            if do_fetch or not tarefas_opcoes:
                with st.spinner("Buscando tarefas do projeto..."):
                    # assinatura segura já existente
                    fetched = get_tasks_by_tag_and_project(
                        proj_id_sel,
                        tag_query=(filtro_tag_cfg or "").strip()
                    ) or []
                    # Normaliza para {id, name}
                    tarefas_opcoes = [{"id": str(t.get("id") or ""), "name": t.get("name") or ""} for t in fetched]
                st.session_state[cache_key] = tarefas_opcoes

            # 4) Aplica filtro por texto no nome (client-side)
            if filtro_txt:
                ft = filtro_txt.lower()
                tarefas_opcoes = [t for t in tarefas_opcoes if ft in (t["name"] or "").lower()]

            tarefas_dict = {f"{t['name']} (#{t['id']})": (t["id"], t["name"]) for t in tarefas_opcoes if t.get("id")}
            st.caption(f"{len(tarefas_dict)} tarefas listadas após filtro")

            # 5) Pré-seleção atual do projeto
            selecionadas_map: dict = (config.tarefas_por_projeto.get(proj_id_sel) or {})
            preselect_labels = [lbl for lbl, (tid, _) in tarefas_dict.items() if tid in selecionadas_map]

            escolhas = st.multiselect(
                "Selecione as tarefas que poderão receber apontamento",
                options=list(tarefas_dict.keys()),
                default=preselect_labels,
                key=K("whitelist_multiselect"),
                help="A seleção fica salva no config. No upload da ficha, usaremos esta lista diretamente."
            )

            colb = st.columns(3)
            with colb[0]:
                if st.button("💾 Salvar seleção para este projeto", type="primary", key=K("whitelist_save_btn")):
                    novo_map = {}
                    for lbl in escolhas:
                        tid, tname = tarefas_dict[lbl]
                        novo_map[tid] = tname
                    config.tarefas_por_projeto[proj_id_sel] = novo_map
                    config_manager.salvar_config()
                    st.success(f"Salvo: {len(novo_map)} tarefas para o projeto #{proj_id_sel}")

            with colb[1]:
                if st.button("🧹 Limpar seleção do projeto", key=K("whitelist_clear_btn")):
                    config.tarefas_por_projeto[proj_id_sel] = {}
                    config_manager.salvar_config()
                    st.info("Seleção do projeto limpa.")

            with colb[2]:
                if st.button("📝 Sincronizar nomes (IDs mantidos)", key=K("whitelist_resync_names_btn")):
                    # Atualiza os nomes salvos conforme últimas opções carregadas
                    cache = {t["id"]: t["name"] for t in tarefas_opcoes}
                    atual = config.tarefas_por_projeto.get(proj_id_sel, {}) or {}
                    for tid in list(atual.keys()):
                        if tid in cache:
                            atual[tid] = cache[tid]
                    config.tarefas_por_projeto[proj_id_sel] = atual
                    config_manager.salvar_config()
                    st.success("Nomes sincronizados a partir da API (IDs inalterados).")
        else:
            st.info("Selecione um projeto para listar e salvar tarefas.")

    # ===== TESTE DE CONFIGURAÇÕES =====
    st.subheader("🧪 Teste de Configurações")
    col_test1, col_test2 = st.columns(2)

    with col_test1:
        if st.button("🔍 Testar Conexão Teamwork", key="test_connection"):
            with st.spinner("Testando conexão..."):
                try:
                    projetos = get_projects_teamwork()
                    if projetos:
                        st.success(f"✅ Conexão OK! Encontrados {len(projetos)} projetos")
                        with st.expander("Ver alguns projetos"):
                            for projeto in projetos[:5]:
                                st.write(f"• **{projeto['name']}** (#{projeto['id']}) - {projeto.get('category','')}")
                    else:
                        st.warning("⚠️ Conexão OK, mas nenhum projeto encontrado")
                except Exception as e:
                    st.error(f"❌ Erro na conexão: {e}")

    with col_test2:
        if st.button("📋 Mostrar Configurações", key="show_config"):
            st.json({
                "tag_teamwork": getattr(config, "tag_teamwork", ""),
                "exigir_consultor": getattr(config, "exigir_consultor", False),
                "ocultar_painel_analise": getattr(config, "ocultar_painel_analise", True),
                "tarefas_por_projeto": {k: list(v.keys()) for k, v in (config.tarefas_por_projeto or {}).items()}
            })

    # ===== INFORMAÇÕES DO SISTEMA =====
    st.subheader("🔗 Informações do Sistema")
    col_sys1, col_sys2 = st.columns(2)

    with col_sys1:
        st.write("**Teamwork:**")
        st.write(f"**URL:** {TEAMWORK_CONFIG['base_url']}")
        st.write(f"**Status:** {'✅ Conectado' if st.session_state.teamwork_client else '❌ Desconectado'}")
        st.write("**APIs Utilizadas:**")
        st.write("• **Projetos:** API v1 (confiável)")
        st.write("• **Tarefas:** API v1 (confiável)")
        st.write("• **Lançamentos:** API v1 (múltiplos formatos)")

    with col_sys2:
        st.write("**Dependências:**")
        st.write(f"• **PDF Processing:** {'✅ pdfplumber disponível' if HAS_PDF else '❌ pdfplumber não instalado'}")
        st.write(f"• **Data Processing:** {'✅ pandas disponível' if HAS_PANDAS else '❌ pandas não instalado'}")
        st.write("**Funcionalidades:**")
        st.write("✅ Extração de fichas Viasell")
        st.write("✅ Interface tabular")
        st.write("✅ Mapeamento de tarefas")
        st.write("✅ Lançamento no Teamwork")
        st.write("✅ Botão final de lançamento")

    if not HAS_PDF or not HAS_PANDAS:
        st.info("💡 Para funcionalidades completas, instale: `pip install pdfplumber pandas`")

if __name__ == "__main__":
    main()
