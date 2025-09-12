"""
Módulo Cliente Teamwork Adaptado para Integração com Sistema ML
Baseado no código original fornecido, adaptado para o sistema de anotações inteligentes
"""

import os
import re
import json
import base64
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union
import requests
import unicodedata


@dataclass
class MetaFicha:
    """Metadados extraídos de uma ficha de serviço"""
    visita: Optional[str] = None
    ticket: Optional[str] = None
    ficha: Optional[str] = None
    cliente: Optional[str] = None


@dataclass
class SessaoTrabalho:
    """Representa uma sessão de trabalho extraída da ficha"""
    data: str        # ISO yyyy-mm-dd
    hora_inicio: str # HH:MM:SS
    hora_fim: str    # HH:MM:SS
    minutos: int
    cobravel: bool
    atividade: str
    
    @property
    def duracao_horas(self) -> float:
        """Retorna a duração em horas decimais"""
        return self.minutos / 60.0


class TeamworkClient:
    """
    Cliente Teamwork com detecção automática de autenticação e funcionalidades
    adaptadas para integração com sistema ML de anotações
    """
    
    def __init__(self, base_url: str, api_key: str, auth_mode: str = "auto"):
        """
        Inicializa o cliente Teamwork
        
        Args:
            base_url: URL base do Teamwork (ex: https://empresa.teamwork.com)
            api_key: Chave da API ou Personal Access Token
            auth_mode: Modo de autenticação ("basic", "bearer", "auto")
        """
        self.base = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.session = requests.Session()
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._auth_mode = (auth_mode or "auto").lower()
        self._active_mode = None

        # cache simples por (project_id, include_completed)
        self._tasks_cache: Dict[Tuple[int, bool], List[Dict[str, Any]]] = {}
        
        # Inicializa como Basic por padrão (mais comum no Teamwork)
        self._set_auth_mode("basic")
    
    def _set_auth_mode(self, mode: str) -> None:
        """Define o modo de autenticação"""
        mode = mode.lower()
        self._active_mode = mode
        
        if mode == "basic":
            token = base64.b64encode(f"{self.api_key}:x".encode("utf-8")).decode("ascii")
            self.headers["Authorization"] = f"Basic {token}"
        elif mode == "bearer":
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            # fallback para basic
            token = base64.b64encode(f"{self.api_key}:x".encode("utf-8")).decode("ascii")
            self.headers["Authorization"] = f"Basic {token}"
            self._active_mode = "basic"
    
    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Executa requisição HTTP com retry automático de autenticação"""
        url = path if path.startswith("http") else f"{self.base}{path}"
        
        # Primeira tentativa com o modo ativo
        resp = self.session.request(method, url, headers=self.headers, timeout=30, **kwargs)
        
        # Se não for auto, apenas retorna/raise
        if self._auth_mode != "auto":
            resp.raise_for_status()
            return resp
        
        # Em auto, se 401, troca o modo e tenta uma vez
        if resp.status_code == 401:
            alt = "bearer" if self._active_mode == "basic" else "basic"
            self._set_auth_mode(alt)
            resp = self.session.request(method, url, headers=self.headers, timeout=30, **kwargs)
        
        resp.raise_for_status()
        return resp
    
    def test_connection(self) -> Dict[str, Any]:
        """Testa a conexão com o Teamwork"""
        try:
            resp = self._request("GET", "/account.json")
            account = resp.json().get("account", {})
            return {
                "success": True,
                "account_name": account.get("name"),
                "auth_mode": self._active_mode,
                "message": f"Conectado com sucesso usando {self._active_mode} auth"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "auth_mode": self._active_mode
            }
    
    def get_projects(self, status: str = "active") -> List[Dict[str, Any]]:
        """
        Lista projetos do Teamwork
        
        Args:
            status: Status dos projetos ("active", "completed", "all")
        """
        params = {}
        if status != "all":
            params["status"] = status
            
        resp = self._request("GET", "/projects.json", params=params)
        projects = resp.json().get("projects", [])
        
        # Enriquecer com informações úteis
        for project in projects:
            project["_display_name"] = f"[{project.get('id')}] {project.get('name', 'Sem nome')}"
            
        return projects
    
    def get_project_by_id(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Obtém um projeto específico por ID"""
        try:
            resp = self._request("GET", f"/projects/{project_id}.json")
            return resp.json().get("project")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def get_tasks_for_project(
        self,
        project_id: int,
        include_completed: bool = True,
        tag_filter: Optional[str] = None,
        page_size: int = 200,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Lista tarefas de um projeto (v1), com suporte a:
          - include_completed: incluir tarefas concluídas
          - tag_filter: filtrar por nome de tag (normalizado)
          - paginação (varrendo pages até esgotar)
          - cache em memória por (project_id, include_completed)
        Retorna as tarefas cruas do Teamwork (todo-items/tasks) enriquecidas.
        """
        cache_key = (int(project_id), bool(include_completed))
        tasks: List[Dict[str, Any]] = []

        # cache
        if use_cache and cache_key in self._tasks_cache and tag_filter is None:
            tasks = self._tasks_cache[cache_key]
        else:
            page = 1
            aggregated: List[Dict[str, Any]] = []
            while True:
                params = {
                    "include": "tags",
                    "pageSize": page_size,
                    "page": page,
                }
                if include_completed:
                    params["includeCompletedTasks"] = "true"

                resp = self._request("GET", f"/projects/{project_id}/tasks.json", params=params)
                data = resp.json()
                # Alguns tenants retornam "todo-items", outros "tasks"
                page_items = data.get("todo-items") or data.get("tasks") or []
                if not page_items:
                    break

                aggregated.extend(page_items)

                # Heurística de parada: se retornou menos que page_size, acabou
                if len(page_items) < page_size:
                    break
                page += 1

            # Enriquecer
            for task in aggregated:
                task["_project_id"] = project_id
                name = task.get("content") or task.get("name") or ""
                task["_display_name"] = f"[{task.get('id')}] {name}"
                task["_tag_names"] = [t.get("name", "") for t in task.get("tags", [])]

            tasks = aggregated
            if use_cache and tag_filter is None:
                self._tasks_cache[cache_key] = tasks

        # Filtrar por tag (se solicitado)
        if tag_filter:
            tag_norm = self._normalize_tag(tag_filter)
            filtered = []
            for t in tasks:
                tag_names_norm = [self._normalize_tag(x) for x in (t.get("_tag_names") or [])]
                name_text = (t.get("content") or t.get("name") or "")
                # aceitar por tag (preferência) ou fallback por ocorrência no nome
                if tag_norm in tag_names_norm or tag_norm in self._normalize_tag(name_text):
                    filtered.append(t)
            tasks = filtered

        return tasks
    
    def get_tasks_by_tag(
        self,
        project_id: int,
        tag_name: str,
        include_completed: bool = True,
        page_size: int = 200,
        use_cache: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Retorna tarefas do projeto que possuem a tag informada, no formato esperado pelo app:
            [{"id": "<id>", "name": "<nome>"}]
        - Usa get_tasks_for_project (com suporte a paginação e cache).
        - Normaliza nome/ID para evitar quebras no select.
        """
        raw_tasks = self.get_tasks_for_project(
            project_id=project_id,
            include_completed=include_completed,
            tag_filter=tag_name,
            page_size=page_size,
            use_cache=use_cache,
        )
        simplified: List[Dict[str, str]] = []
        for t in raw_tasks:
            tid = str(t.get("id") or t.get("id_str") or "")
            name = t.get("content") or t.get("name") or ""
            simplified.append({"id": tid, "name": name})
        return simplified

    def get_task_by_id(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Obtém uma tarefa específica por ID"""
        try:
            resp = self._request("GET", f"/tasks/{task_id}.json")
            return resp.json().get("todo-item")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def list_time_entries(self, task_id: int, date_iso: str) -> List[Dict[str, Any]]:
        """Lista lançamentos de horas existentes para uma tarefa em uma data"""
        ymd = date_iso.replace("-", "")
        params = {
            "taskId": task_id,
            "fromDate": ymd,
            "toDate": ymd
        }
        
        try:
            resp = self._request("GET", "/time_entries.json", params=params)
            return resp.json().get("time-entries", [])
        except requests.HTTPError as e:
            if e.response.status_code >= 400:
                return []
            raise
    
    def create_time_entry(self, 
                         task_id: Optional[int] = None,
                         project_id: Optional[int] = None,
                         date_iso: str = None,
                         minutes: int = 0,
                         description: str = "",
                         is_billable: bool = True) -> Dict[str, Any]:
        """
        Cria um lançamento de horas no Teamwork
        
        Args:
            task_id: ID da tarefa (opcional se project_id fornecido)
            project_id: ID do projeto (opcional se task_id fornecido)
            date_iso: Data no formato ISO (YYYY-MM-DD)
            minutes: Duração em minutos
            description: Descrição do trabalho realizado
            is_billable: Se o tempo é cobrável
        """
        if not (task_id or project_id):
            raise ValueError("Informe task_id ou project_id para o lançamento")
        
        if not date_iso:
            date_iso = dt.date.today().isoformat()
        
        payload = {
            "time-entry": {
                "description": description[:4000],  # Limite do Teamwork
                "date": date_iso,
                "minutes": int(minutes),
                "isBillable": bool(is_billable),
            }
        }
        
        if task_id:
            payload["time-entry"]["taskId"] = int(task_id)
        elif project_id:
            payload["time-entry"]["projectId"] = int(project_id)
        
        resp = self._request("POST", "/time_entries.json", json=payload)
        return resp.json()
    
    def check_duplicate_entry(self, task_id: int, date_iso: str, 
                            description: str, minutes: int) -> bool:
        """
        Verifica se já existe um lançamento idêntico
        
        Returns:
            True se encontrar duplicata, False caso contrário
        """
        existing = self.list_time_entries(task_id, date_iso)
        
        for entry in existing:
            if (entry.get("description", "").strip() == description.strip() and 
                int(entry.get("minutes", 0)) == minutes):
                return True
        
        return False
    
    def _normalize_tag(self, tag: str) -> str:
        """Normaliza uma tag para comparação: lower + sem acentos + espaçamentos unificados"""
        if tag is None:
            return ""
        tag = self._strip_accents(tag.lower())
        tag = re.sub(r"[\\/:]", " - ", tag)
        tag = re.sub(r"\s+", " ", tag).strip()
        return tag
    
    def suggest_task_for_activity(self, 
                                activity_text: str,
                                phase: str,
                                meta: MetaFicha,
                                tasks: List[Dict[str, Any]]) -> Optional[int]:
        """
        Sugere uma tarefa baseada na atividade descrita
        
        Args:
            activity_text: Texto da atividade
            phase: Fase do projeto
            meta: Metadados da ficha
            tasks: Lista de tarefas disponíveis
        Returns:
            ID da tarefa sugerida ou None
        """
        activity_tokens = set(self._tokenize(activity_text))
        phase_tokens = set(self._tokenize(phase))
        client_tokens = set(self._tokenize(meta.cliente or ""))
        
        all_tokens = activity_tokens | phase_tokens | client_tokens
        
        best_task_id = None
        best_score = 0
        
        for task in tasks:
            task_id = task.get("id")
            task_name = task.get("content") or task.get("name") or ""
            task_tags = " ".join(task.get("_tag_names", []))
            
            task_tokens = set(self._tokenize(task_name + " " + task_tags))
            
            # Score base: interseção de tokens
            score = len(all_tokens & task_tokens)
            
            # Bônus por ticket/ficha no nome
            if meta.ticket and re.search(re.escape(str(meta.ticket)), task_name, re.IGNORECASE):
                score += 3
            
            if meta.ficha and re.search(re.escape(str(meta.ficha)), task_name, re.IGNORECASE):
                score += 2
            
            # Bônus se fase aparece no nome/tag
            if len(phase_tokens & task_tokens) > 0:
                score += 1
            
            if score > best_score:
                best_score = score
                best_task_id = int(task_id)
        
        return best_task_id if best_score > 0 else None
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokeniza um texto para análise de similaridade"""
        if not text:
            return []
        
        # Remove acentos
        text = self._strip_accents(text.lower())
        
        # Remove caracteres especiais, mantém apenas letras e números
        text = re.sub(r"[^a-z0-9]+", " ", text)
        
        # Retorna tokens com mais de 2 caracteres
        return [token for token in text.split() if len(token) > 2]
    
    def _strip_accents(self, text: str) -> str:
        """Remove acentos de um texto"""
        return "".join(
            char for char in unicodedata.normalize("NFD", text)
            if unicodedata.category(char) != "Mn"
        )
    
    def format_time_entry_description(self, 
                                    phase: str,
                                    meta: MetaFicha,
                                    sessao: SessaoTrabalho,
                                    include_ml_info: bool = False,
                                    ml_analysis: Optional[Dict] = None) -> str:
        """
        Formata a descrição do lançamento de horas
        """
        # Cabeçalho com informações da ficha
        header = f"[{phase} | Ficha {meta.ficha or '-'} | Ticket {meta.ticket or '-'}]"
        
        # Período trabalhado
        period = f"Período: {sessao.hora_inicio}–{sessao.hora_fim}"
        
        # Descrição da atividade
        activity = (sessao.atividade or "").strip()
        
        # Informações do ML (se solicitado)
        ml_info = ""
        if include_ml_info and ml_analysis:
            ml_parts = []
            
            if ml_analysis.get("tipo_servico_sugerido"):
                ml_parts.append(f"Tipo: {ml_analysis['tipo_servico_sugerido']}")
            
            if ml_analysis.get("palavras_chave"):
                keywords = ", ".join(ml_analysis["palavras_chave"][:5])
                ml_parts.append(f"Palavras-chave: {keywords}")
            
            if ml_analysis.get("confianca"):
                confidence = ml_analysis["confianca"]
                ml_parts.append(f"Confiança ML: {confidence:.1%}")
            
            if ml_parts:
                ml_info = f"\n[ML] {' | '.join(ml_parts)}"
        
        # Combina todas as partes
        parts = [header, period, activity, ml_info]
        return "\n".join(part for part in parts if part.strip())
    
    def create_fingerprint(self, meta: MetaFicha, sessao: SessaoTrabalho, 
                          task_id: int) -> str:
        """Cria uma impressão digital única para evitar duplicatas"""
        base = f"{meta.ficha}|{meta.ticket}|{sessao.data}|{sessao.hora_inicio}|{sessao.hora_fim}|{task_id}"
        return hashlib.sha256(base.encode()).hexdigest()[:16]


class TeamworkIntegrationError(Exception):
    """Exceção específica para erros de integração com Teamwork"""
    pass


def create_teamwork_client_from_config(config: Dict[str, Any]) -> TeamworkClient:
    """
    Cria um cliente Teamwork a partir de configuração
    """
    required_keys = ["base_url", "api_key"]
    missing_keys = [key for key in required_keys if not config.get(key)]
    
    if missing_keys:
        raise TeamworkIntegrationError(f"Configuração incompleta. Chaves faltando: {missing_keys}")
    
    return TeamworkClient(
        base_url=config["base_url"],
        api_key=config["api_key"],
        auth_mode=config.get("auth_mode", "auto")
    )


def validate_teamwork_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida configuração do Teamwork
    """
    try:
        client = create_teamwork_client_from_config(config)
        return client.test_connection()
    except Exception as e:
        return {
            "success": False,
            "error": f"Erro na validação: {str(e)}"
        }
