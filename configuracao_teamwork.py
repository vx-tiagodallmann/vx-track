"""
Configuração do Teamwork baseada nas informações fornecidas pelo usuário
"""

# Configurações de conexão Teamwork
TEAMWORK_CONFIG = {
    "base_url": "https://viasoft.teamwork.com",
    "api_key": "twp_g8b0t3nBOrRSoBK2RYCZ5XTCF9ZS",
    "auth_mode": "basic"
}

# Tag para identificar tarefas que recebem apontamento de horas
TAG_APONTAMENTO = "Apontável"

# IDs das categorias alvo no Teamwork
CATEGORIAS_ALVO = [
    20649,  # 4.1 CONSTRUSHOW MATRIZ
    20526,  # 4.1.1 CANDS
    20835,  # 4.1.1 EXECUÇÃO
    20527,  # 4.1.2 GO UP
    20834,  # 4.1.2 PÓS GO-LIVE
    20651,  # 4.1.3 LZ
    20833,  # 4.1.3 PARALISADO
    20650,  # 4.2 CONSTRUSHOW FRANQUIAS
]

# Fases fixas do projeto (tarefas que recebem apontamento)
FASES_PROJETO = [
    "Análise de Processos/ Aderência",
    "Mapeamento Operacional", 
    "1° Importação de Dados",
    "Configurações",
    "Tributação",
    "Validação das Configurações",
    "Treinamento a Usuários",
    "Simulação e Homologação",
    "Preparação Go-Live",
    "Go-Live/ Operação Assistida",
    "Treinar Supervisor do Cliente",
    "Retorno Técnico - Pós Go-Live",
    "Fechamentos"
]

# Mapeamento de tipos de serviço ML para fases do projeto
MAPEAMENTO_ML_FASES = {
    "Implantação": ["1° Importação de Dados", "Preparação Go-Live", "Go-Live/ Operação Assistida"],
    "Configuração": ["Configurações", "Tributação", "Validação das Configurações"],
    "Treinamento": ["Treinamento a Usuários", "Treinar Supervisor do Cliente"],
    "Suporte": ["Retorno Técnico - Pós Go-Live", "Fechamentos"],
    "Personalização": ["Análise de Processos/ Aderência", "Mapeamento Operacional"],
    "Outros": ["Simulação e Homologação"]
}

# Configurações de ML
ML_CONFIG = {
    "incluir_analise_ml_descricao": True,
    "max_palavras_chave": 5,
    "incluir_confianca": True,
    "threshold_confianca_minima": 0.7
}

# Configurações de interface
UI_CONFIG = {
    "permitir_gravacao_audio": True,
    "validacao_obrigatoria": True,
    "dry_run_padrao": True,
    "verificar_duplicatas": True
}

