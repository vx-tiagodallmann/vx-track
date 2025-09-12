import streamlit as st
import base64
import json
import os
import requests
import uuid
from dataclasses import dataclass, asdict, fields
from datetime import datetime, time
from io import BytesIO
from typing import Any, Dict, List

# Importar módulos existentes
from configuracao_teamwork import TEAMWORK_CONFIG
from ml_utils import MLAnalyzer
from teamwork_client import TeamworkClient


# =========================
#  Util / Namespace p/ keys
# =========================
NVF_NS = "nvf"  # namespace desta página
def K(name: str) -> str:
    return f"{NVF_NS}:{name}"


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


# =========================
#  Helpers genéricos
# =========================
def _filter_keys(d: dict, data_cls):
    """Retorna apenas as chaves de d que existem como campos do dataclass data_cls."""
    allowed = {f.name for f in fields(data_cls)}
    return {k: v for k, v in (d or {}).items() if k in allowed}


# =========================
#  PDF
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

def load_all_configs():
    return {
        'teamwork_api_key': os.getenv("TEAMWORK_API_KEY"),
        'teamwork_base_url': os.getenv("TEAMWORK_BASE_URL"),
        'viahelper_upload_url': os.getenv("VIAHELPER_UPLOAD_URL"),
        'viahelper_api_key': os.getenv("VIAHELPER_API_KEY"),
        'assistant_id': os.getenv("ASSISTANT_ID")
    }

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

def _teamwork_auth_headers():
    token = base64.b64encode(f"{TEAMWORK_CONFIG['api_key']}:x".encode()).decode()
    return {"Authorization": f"Basic {token}", "Accept": "application/json"}

def get_tasks_by_tag_fallback(project_id: str, tag_query: str) -> List[Dict[str, str]]:
    base = TEAMWORK_CONFIG['base_url'].rstrip('/')
    url = f"{base}/projects/{project_id}/tasks.json?include=tags&pageSize=200"
    try:
        resp = requests.get(url, headers=_teamwork_auth_headers(), timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.warning(f"Não foi possível consultar tarefas no Teamwork: {e}")
        return []
    items = data.get("tasks") or data.get("todo-items") or []
    tag_q = (tag_query or "").strip().lower()
    out: List[Dict[str, str]] = []
    for it in items:
        name = it.get("content") or it.get("name") or ""
        tid = str(it.get("id") or it.get("id_str") or "")
        tag_names = [t.get("name", "").lower() for t in it.get("tags", [])]
        if tag_q and (tag_q in tag_names or tag_q in name.lower()):
            out.append({"id": tid, "name": name})
    return out

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
# =========================
#  App
# =========================
def main():

    st.set_page_config(
        page_title="Sistema de Anotações Inteligentes",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🤖 Sistema de Anotações Inteligentes")
    st.markdown("**Versão com Registros Dinâmicos e Integração Teamwork**")

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
    if "ficha_atual" not in st.session_state:
        st.session_state.ficha_atual = None
    if "registros_temp" not in st.session_state:
        st.session_state.registros_temp = []

    # Sidebar
    with st.sidebar:
        st.title("📋 Navegação")
        opcao = st.selectbox(
            "Escolha uma opção:",
            ["Nova Ficha de Serviço", "Fichas em Andamento", "Histórico de Fichas", "Configurações"],
            key=K("menu")
        )

    if opcao == "Nova Ficha de Serviço":
        nova_ficha_servico()
    elif opcao == "Fichas em Andamento":
        fichas_em_andamento()
    elif opcao == "Histórico de Fichas":
        historico_fichas()
    elif opcao == "Configurações":
        configuracoes()


def nova_ficha_servico():
    st.header("📝 Nova Ficha de Serviço")

    # ---------------------------------------------- Seção 1: Campos Fixos --------------------------------------
    st.subheader("📋 Informações Básicas")
    c1, c2 = st.columns(2)

    with c1:
        st.write("**Cliente:**")
        projetos = st.session_state.teamwork_client.get_projects()
        if projetos:
            projeto_opcoes = {f"{p['name']} (ID: {p['id']})": p for p in projetos}
            projeto_label = st.selectbox(
                "Selecione o projeto:",
                options=list(projeto_opcoes.keys()),
                key=K("projeto_select")
            )
            if projeto_label:
                projeto = projeto_opcoes[projeto_label]
                cliente = projeto.get("name", "")
                projeto_id = str(projeto.get("id"))
            else:
                cliente, projeto_id = "", ""
        else:
            cliente = st.text_input("Nome do Cliente:", key=K("cliente_manual"))
            projeto_id = "manual"

        st.write("**Vertical:**")
        vertical = st.selectbox(
            "Selecione a vertical:",
            ["Agrotitan", "Construshow", "Petroshow"],
            key=K("vertical_select")
        )

    with c2:
        st.write("**Tipo de Serviço:**")
        tipo_servico = st.selectbox(
            "Selecione o tipo:",
            ["Implantação", "Personalização", "Serviços", "Deslocamento", "Outros"],
            key=K("tipo_servico_select")
        )
        st.write("**Valor da Hora:**")
        valor_hora = st.number_input(
            "R$ por hora:",
            min_value=0.0,
            value=180.0,
            step=10.0,
            key=K("valor_hora_input")
        )

    st.divider()
    # Botão para iniciar nova ficha
    if st.session_state.ficha_atual is None:
        if st.button("🚀 Iniciar Nova Ficha", type="primary", key=K("btn_iniciar")):
            if cliente:
                ficha_id = st.session_state.ficha_manager.criar_ficha(
                    cliente, projeto_id, vertical, tipo_servico, valor_hora
                )
                st.session_state.ficha_atual = ficha_id
                st.session_state.registros_temp = []
                st.success(f"Ficha {ficha_id} criada com sucesso!")
                st.rerun()
            else:
                st.error("Por favor, selecione um cliente/projeto válido.")
        return

    st.info(f"📋 Ficha Atual: {st.session_state.ficha_atual}")
    st.subheader("⏱️ Registros de Atividades")

    # -------------------------------- FORMULÁRIO DE REGISTRO --------------------------------------
    form = st.form(K("novo_registro"), clear_on_submit=False)
    form.write("**Novo Registro de Atividade:**")

    f1, f2, f3, f4 = form.columns(4)

    # COL 1 — Data + Executor (Teamwork)
    with f1:
        form.date_input("Data:", key=K("data_registro"))

        # Pessoas do projeto (Teamwork)
        try:
            pessoas = get_project_people_fallback(projeto_id)
        except Exception:
            pessoas = []

        exec_map = {f"{p['name']} (ID: {p['id']})": {"id": str(p["id"]), "name": p["name"]} for p in pessoas}
        st.session_state[K("_exec_map")] = exec_map  # <-- guarda para o callback

        exec_options = [""] + list(exec_map.keys()) + ["Outro (digitar)"]
        escolha = form.selectbox("Executor (Nome):", options=exec_options, key=K("executor_select"))

        if escolha == "Outro (digitar)":
            form.text_input("Digite o executor:", key=K("executor_manual"))
        else:
            st.session_state[K("executor_manual")] = ""

    # COL 2 — Horários
    with f2:
        form.time_input("Hr Início:", key=K("hr_inicio_input"))
        form.time_input("Hr Fim:", key=K("hr_fim_input"))

    # COL 3 — Total + Faturável
    with f3:
        hi = st.session_state.get(K("hr_inicio_input"))
        hf = st.session_state.get(K("hr_fim_input"))
        total_vis = "00:00:00"
        if hi and hf:
            total_vis = calcular_total_horas(hi.strftime("%H:%M"), hf.strftime("%H:%M"))
        form.text_input("Total de Hrs:", value=total_vis, key=K("total_hrs_display"), disabled=True)
        form.checkbox("Faturável", value=st.session_state.get(K("faturavel_input"), True), key=K("faturavel_input"))

    # COL 4 — Tag + Tarefa
    with f4:
        tag_apontamento = form.text_input("Tag para consulta:", value="apontavel", key=K("tag_input"))
        if K("_tarefa_opcoes_map") not in st.session_state:
            st.session_state[K("_tarefa_opcoes_map")] = {}

        tarefas = []
        if tag_apontamento and projeto_id and projeto_id != "manual":
            try:
                proj_id_int = int(projeto_id)
            except Exception:
                proj_id_int = projeto_id
            try:
                tarefas = st.session_state.teamwork_client.get_tasks_by_tag(proj_id_int, tag_apontamento)
            except Exception:
                try:
                    tarefas = get_tasks_by_tag_fallback(proj_id_int, tag_apontamento)
                except Exception:
                    tarefas = []

        if tarefas:
            tarefa_opcoes = {
                f"{t['name']} (ID: {t['id']})": {"id": str(t["id"]), "name": t["name"]}
                for t in tarefas
            }
            st.session_state[K("_tarefa_opcoes_map")] = tarefa_opcoes
            form.selectbox("Tarefa:", options=[""] + list(tarefa_opcoes.keys()), key=K("tarefa_select"))
        else:
            st.session_state[K("_tarefa_opcoes_map")] = {}
            st.session_state[K("tarefa_select")] = ""
            form.caption("Nenhuma tarefa encontrada com essa tag (ou projeto manual).")

    # Callback de insert
    def _on_inserir_registro():
        s = st.session_state
        
        # Deriva executor a partir dos widgets
        exec_sel = s.get(K("executor_select"), "")
        exec_map = s.get(K("_exec_map"), {}) or {}

        if exec_sel == "Outro (digitar)":
            executor_val = (s.get(K("executor_manual")) or "").strip()
            executor_id = ""  # sem ID quando é manual
        elif exec_sel:
            info = exec_map.get(exec_sel, {})
            executor_val = (info.get("name") or "").strip()
            executor_id = (str(info.get("id") or "")).strip()
        else:
            executor_val, executor_id = "", ""
        
        # >>> Se o person-id tem que ser obrigatório, ative esta checagem:
        if not executor_id:
            s[K("_form_error")] = "Selecione um executor da lista (com ID do Teamwork)."
            return
        
        obrigatorios = [
            s.get(K("data_registro")),
            executor_val,
            s.get(K("hr_inicio_input")),
            s.get(K("hr_fim_input")),
            s.get(K("descricao_input")),
        ]
        if not all(obrigatorios):
            s[K("_form_error")] = "Por favor, preencha todos os campos obrigatórios (incluindo o Executor)."
            return


        total_hrs = calcular_total_horas(
            s[K("hr_inicio_input")].strftime("%H:%M"),
            s[K("hr_fim_input")].strftime("%H:%M"),
        )

        tarefa_opcoes = s.get(K("_tarefa_opcoes_map"), {}) or {}
        tarefa_sel = s.get(K("tarefa_select"), "")
        tarefa_id = tarefa_opcoes.get(tarefa_sel, {}).get("id", "") if tarefa_sel else ""
        tarefa_nome = tarefa_sel or ""

        registro = RegistroAtividade(
            id=str(uuid.uuid4()),
            data=s[K("data_registro")].strftime("%d/%m/%Y"),
            executor=executor_val,
            executor_id=executor_id,
            hr_inicio=s[K("hr_inicio_input")].strftime("%H:%M:%S"),
            hr_fim=s[K("hr_fim_input")].strftime("%H:%M:%S"),
            descricao=s[K("descricao_input")],
            total_hrs=total_hrs,
            faturavel=s.get(K("faturavel_input"), True),
            tarefa_id=tarefa_id,
            tarefa_nome=tarefa_nome,
            ml_sugestao=s.get("ml_sugestao", ""),
            ml_confianca=s.get("ml_confianca", 0.0),
        )

        s.ficha_manager.adicionar_registro(s.ficha_atual, registro)
        s.registros_temp.append(registro)

        # reset widgets
        s[K("data_registro")] = datetime.now().date()
        s[K("hr_inicio_input")] = time(0, 0)
        s[K("hr_fim_input")] = time(0, 0)
        s[K("descricao_input")] = ""
        s[K("tag_input")] = "apontavel"
        s[K("tarefa_select")] = ""
        s[K("faturavel_input")] = True
        s[K("executor_select")] = ""
        s[K("executor_manual")] = ""
        s[K("executor_input")] = ""
        s[K("executor_id")] = ""

        s[K("_form_success")] = "Registro adicionado com sucesso!"
        
    # Campo de descrição e Botão de inserir
    form.text_area(
        "Descrição da atividade:",
        value=st.session_state.get("descricao_transcrita", ""),
        height=100,
        key=K("descricao_input"),
    )
    form.form_submit_button("➕ Inserir Registro", type="primary", use_container_width=True, on_click=_on_inserir_registro)

    # Feedback do form
    ok = st.session_state.pop(K("_form_success"), None)
    er = st.session_state.pop(K("_form_error"), None)
    if ok:
        st.success(ok)
    if er:
        st.error(er)

    # ---------- Registros adicionados ----------
    if st.session_state.registros_temp:
        st.subheader("📋 Registros Adicionados")
        for i, registro in enumerate(st.session_state.registros_temp):
            with st.expander(f"Registro {i+1}: {registro.data} - {registro.executor}"):
                cA, cB = st.columns(2)
                with cA:
                    st.write(f"**Data:** {registro.data}")
                    st.write(f"**Executor:** {registro.executor}")
                    st.write(f"**Horário:** {registro.hr_inicio} - {registro.hr_fim}")
                    st.write(f"**Total:** {registro.total_hrs}")
                with cB:
                    st.write(f"**Faturável:** {'Sim' if registro.faturavel else 'Não'}")
                    st.write(f"**Tarefa:** {registro.tarefa_nome or '-'}")
                    if registro.ml_sugestao:
                        st.write(f"**ML Sugestão:** {registro.ml_sugestao} ({registro.ml_confianca:.1%})")
                st.write(f"**Descrição:** {registro.descricao}")

    # ---------- Concluir ficha ----------
    st.divider()
    b1, b2, b3 = st.columns([1, 1, 1])
    with b2:
        if st.button("🏁 Concluir Ficha", type="primary", key=K("concluir_ficha")):
            ok_ct, fail_ct = 0, 0
            erros: list[str] = []

            # Lança cada registro no Teamwork
            try:
                ficha = st.session_state.ficha_manager.fichas.get(st.session_state.ficha_atual)
                if ficha and ficha.registros:
                    for reg in ficha.registros:
                        try:
                            _post_time_entry_tw(reg, ficha.projeto_id)
                            ok_ct += 1
                        except requests.HTTPError as e:
                            fail_ct += 1
                            msg = getattr(e.response, "text", str(e))
                            erros.append(f"Falha ao lançar '{reg.executor}' em {reg.data} ({reg.hr_inicio}): {e} | {msg}")
                        except Exception as e:
                            fail_ct += 1
                            erros.append(f"Falha ao lançar '{reg.executor}' em {reg.data} ({reg.hr_inicio}): {e}")

                st.info(f"Lançamentos no Teamwork: ✅ {ok_ct} | ❌ {fail_ct}")
                for m in erros:
                    st.warning(m)

            except Exception as e:
                st.error(f"Erro geral ao lançar apontamentos: {e}")

            # Gera e oferece o PDF
            pdf_bytes = st.session_state.ficha_manager.concluir_ficha(st.session_state.ficha_atual)
            if pdf_bytes:
                st.download_button(
                    label="📄 Download PDF da Ficha",
                    data=pdf_bytes,
                    file_name=f"{st.session_state.ficha_atual}.pdf",
                    mime="application/pdf",
                    key=K("download_pdf"),
                )
                st.success("Ficha concluída com sucesso!")
                st.info("PDF gerado para lançamento no Viasell")
                st.session_state.ficha_atual = None
                st.session_state.registros_temp = []
            else:
                st.error("Erro ao gerar PDF da ficha")


def fichas_em_andamento():
    st.header("📋 Fichas em Andamento")
    fichas_andamento = {k: v for k, v in st.session_state.ficha_manager.fichas.items() if v.status == "Em Andamento"}
    if fichas_andamento:
        for ficha_id, ficha in fichas_andamento.items():
            with st.expander(f"Ficha {ficha_id} - {ficha.cliente}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Cliente:** {ficha.cliente}")
                    st.write(f"**Vertical:** {ficha.vertical}")
                    st.write(f"**Tipo:** {ficha.tipo_servico}")
                with c2:
                    st.write(f"**Valor/Hora:** R$ {ficha.valor_hora:.2f}")
                    st.write(f"**Registros:** {len(ficha.registros)}")
                    st.write(f"**Criada:** {ficha.data_criacao[:10]}")
    else:
        st.info("Nenhuma ficha em andamento")


def historico_fichas():
    st.header("📚 Histórico de Fichas")
    fichas_concluidas = {k: v for k, v in st.session_state.ficha_manager.fichas.items() if v.status == "Concluída"}
    if fichas_concluidas:
        for ficha_id, ficha in fichas_concluidas.items():
            with st.expander(f"Ficha {ficha_id} - {ficha.cliente} (Concluída)"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Cliente:** {ficha.cliente}")
                    st.write(f"**Vertical:** {ficha.vertical}")
                    st.write(f"**Tipo:** {ficha.tipo_servico}")
                with c2:
                    st.write(f"**Valor/Hora:** R$ {ficha.valor_hora:.2f}")
                    st.write(f"**Registros:** {len(ficha.registros)}")
                    st.write(f"**Criada:** {ficha.data_criacao[:10]}")
                if ficha.registros:
                    st.write("**Registros:**")
                    for registro in ficha.registros:
                        st.write(f"- {registro.data}: {registro.executor} ({registro.total_hrs})")
    else:
        st.info("Nenhuma ficha concluída")


def configuracoes():
    st.header("⚙️ Configurações")
    st.subheader("🔗 Teamwork")
    st.write(f"**URL:** {TEAMWORK_CONFIG['base_url']}")
    st.write(f"**Status:** {'Conectado' if st.session_state.teamwork_client else 'Desconectado'}")
    st.subheader("🤖 Machine Learning")
    st.write("**Modelo:** Random Forest (68% acurácia)")
    st.write("**Status:** Ativo")
    st.subheader("🎤 Transcrição de Áudio")
    st.write("**Status:** Desativado no momento")


if __name__ == "__main__":
    main()
