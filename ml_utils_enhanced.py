"""
Módulo ML Aprimorado com Modelo Treinado
Integra o modelo Random Forest treinado com dados históricos
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import re
import unicodedata
from datetime import datetime
import os

class EnhancedMLAnalyzer:
    """Analisador ML aprimorado com modelo treinado em dados históricos"""
    
    def __init__(self, model_path: str = '/home/ubuntu/best_ml_model.pkl'):
        """
        Inicializa o analisador com modelo treinado
        
        Args:
            model_path: Caminho para o arquivo do modelo treinado
        """
        self.model_path = model_path
        self.model = None
        self.model_name = None
        self.training_date = None
        self.confidence_threshold = 0.6
        
        # Carregar modelo se existir
        self.load_model()
        
        # Fallback para regras baseadas em palavras-chave
        self.keyword_rules = {
            'Configuração': [
                'configuração', 'configurar', 'config', 'parametrizar', 'parametros',
                'setup', 'instalação', 'instalar', 'certificado', 'nfe', 'nfce',
                'fiscal', 'tributação', 'impostos', 'icms', 'ipi', 'pis', 'cofins'
            ],
            'Treinamento': [
                'treinamento', 'treinar', 'capacitação', 'capacitar', 'ensinar',
                'explicar', 'demonstrar', 'orientar', 'instruir', 'curso',
                'apresentação', 'workshop', 'tutorial'
            ],
            'Suporte': [
                'suporte', 'ajuda', 'problema', 'erro', 'bug', 'falha',
                'dúvida', 'questão', 'dificuldade', 'resolver', 'solucionar',
                'corrigir', 'debug', 'troubleshooting'
            ],
            'Implantação': [
                'implantação', 'implantar', 'implementação', 'implementar',
                'deploy', 'migração', 'migrar', 'conversão', 'converter',
                'go-live', 'golive', 'produção', 'ativação'
            ]
        }
    
    def load_model(self) -> bool:
        """
        Carrega o modelo treinado
        
        Returns:
            True se modelo foi carregado com sucesso, False caso contrário
        """
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.model_name = model_data['model_name']
                self.training_date = model_data.get('training_date')
                
                print(f"✅ Modelo carregado: {self.model_name}")
                print(f"📅 Data de treinamento: {self.training_date}")
                return True
            else:
                print(f"⚠️ Modelo não encontrado em: {self.model_path}")
                print("🔄 Usando classificação baseada em regras como fallback")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {str(e)}")
            print("🔄 Usando classificação baseada em regras como fallback")
            return False
    
    def preprocess_text(self, text: str) -> str:
        """
        Pré-processa texto da mesma forma que no treinamento
        
        Args:
            text: Texto a ser processado
            
        Returns:
            Texto pré-processado
        """
        if not text or pd.isna(text):
            return ""
        
        # Converter para string e minúsculas
        text = str(text).lower()
        
        # Remover acentos
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        
        # Remover caracteres especiais, manter apenas letras, números e espaços
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remover espaços múltiplos
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def classify_with_model(self, text: str) -> Tuple[str, float]:
        """
        Classifica texto usando modelo treinado
        
        Args:
            text: Texto a ser classificado
            
        Returns:
            Tupla (tipo_servico, confianca)
        """
        if not self.model:
            return self.classify_with_rules(text)
        
        try:
            # Pré-processar texto
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return "Outros", 0.0
            
            # Fazer predição
            prediction = self.model.predict([processed_text])[0]
            
            # Obter probabilidades se disponível
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba([processed_text])[0]
                confidence = max(probabilities)
                
                # Se confiança muito baixa, usar regras como fallback
                if confidence < self.confidence_threshold:
                    rule_type, rule_conf = self.classify_with_rules(text)
                    if rule_conf > 0.5:  # Se regras têm boa confiança
                        return rule_type, rule_conf
            else:
                confidence = 0.8  # Confiança padrão para modelos sem probabilidade
            
            return prediction, confidence
            
        except Exception as e:
            print(f"❌ Erro na classificação com modelo: {str(e)}")
            return self.classify_with_rules(text)
    
    def classify_with_rules(self, text: str) -> Tuple[str, float]:
        """
        Classifica texto usando regras baseadas em palavras-chave (fallback)
        
        Args:
            text: Texto a ser classificado
            
        Returns:
            Tupla (tipo_servico, confianca)
        """
        if not text or pd.isna(text):
            return "Outros", 0.0
        
        text_lower = text.lower()
        scores = {}
        
        # Calcular score para cada categoria
        for category, keywords in self.keyword_rules.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            
            if score > 0:
                # Normalizar score pelo número de palavras-chave da categoria
                scores[category] = score / len(keywords)
        
        if not scores:
            return "Outros", 0.3
        
        # Retornar categoria com maior score
        best_category = max(scores, key=scores.get)
        confidence = min(scores[best_category] * 2, 1.0)  # Normalizar para [0,1]
        
        return best_category, confidence
    
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extrai palavras-chave relevantes do texto
        
        Args:
            text: Texto para extração
            max_keywords: Número máximo de palavras-chave
            
        Returns:
            Lista de palavras-chave
        """
        if not text or pd.isna(text):
            return []
        
        # Pré-processar texto
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return []
        
        # Palavras comuns a ignorar
        stop_words = {
            'de', 'da', 'do', 'das', 'dos', 'a', 'o', 'as', 'os', 'e', 'ou',
            'para', 'por', 'com', 'em', 'no', 'na', 'nos', 'nas', 'um', 'uma',
            'uns', 'umas', 'foi', 'ser', 'ter', 'que', 'se', 'ao', 'aos',
            'realizada', 'realizado', 'feito', 'feita', 'executado', 'executada'
        }
        
        # Extrair palavras
        words = processed_text.split()
        
        # Filtrar palavras relevantes
        keywords = []
        for word in words:
            if (len(word) >= 3 and 
                word not in stop_words and 
                not word.isdigit()):
                keywords.append(word)
        
        # Remover duplicatas mantendo ordem
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            if keyword not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword)
        
        return unique_keywords[:max_keywords]
    
    def generate_summary(self, text: str, max_length: int = 100) -> str:
        """
        Gera resumo do texto
        
        Args:
            text: Texto para resumir
            max_length: Comprimento máximo do resumo
            
        Returns:
            Resumo do texto
        """
        if not text or pd.isna(text):
            return ""
        
        # Se texto já é curto, retornar como está
        if len(text) <= max_length:
            return text
        
        # Dividir em sentenças
        sentences = re.split(r'[.!?]+', text)
        
        if not sentences:
            return text[:max_length] + "..."
        
        # Pegar primeira sentença completa que caiba no limite
        summary = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                if len(summary + sentence) <= max_length:
                    summary += sentence + ". "
                else:
                    break
        
        # Se nenhuma sentença coube, truncar primeira sentença
        if not summary and sentences[0]:
            summary = sentences[0].strip()[:max_length-3] + "..."
        
        return summary.strip()
    
    def analyze_text(self, text: str, include_ml_analysis: bool = True) -> Dict[str, Any]:
        """
        Análise completa do texto
        
        Args:
            text: Texto a ser analisado
            include_ml_analysis: Se deve incluir análise ML
            
        Returns:
            Dicionário com resultados da análise
        """
        if not text or pd.isna(text):
            return {
                'tipo_servico': 'Outros',
                'confianca': 0.0,
                'palavras_chave': [],
                'resumo': '',
                'modelo_usado': 'Nenhum',
                'timestamp': datetime.now().isoformat()
            }
        
        # Classificação
        if include_ml_analysis:
            tipo_servico, confianca = self.classify_with_model(text)
            modelo_usado = self.model_name if self.model else 'Regras'
        else:
            tipo_servico, confianca = self.classify_with_rules(text)
            modelo_usado = 'Regras'
        
        # Extração de palavras-chave
        palavras_chave = self.extract_keywords(text)
        
        # Geração de resumo
        resumo = self.generate_summary(text)
        
        return {
            'tipo_servico': tipo_servico,
            'confianca': round(confianca, 3),
            'palavras_chave': palavras_chave,
            'resumo': resumo,
            'modelo_usado': modelo_usado,
            'timestamp': datetime.now().isoformat(),
            'texto_original': text,
            'comprimento_texto': len(text)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo carregado
        
        Returns:
            Dicionário com informações do modelo
        """
        return {
            'modelo_carregado': self.model is not None,
            'nome_modelo': self.model_name,
            'data_treinamento': self.training_date,
            'caminho_modelo': self.model_path,
            'threshold_confianca': self.confidence_threshold,
            'fallback_disponivel': True
        }
    
    def update_confidence_threshold(self, new_threshold: float) -> None:
        """
        Atualiza o threshold de confiança
        
        Args:
            new_threshold: Novo threshold (0.0 a 1.0)
        """
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            print(f"✅ Threshold de confiança atualizado para: {new_threshold}")
        else:
            print("❌ Threshold deve estar entre 0.0 e 1.0")


# Instância global para uso na aplicação
enhanced_analyzer = EnhancedMLAnalyzer()


def analyze_service_description(text: str, **kwargs) -> Dict[str, Any]:
    """
    Função de conveniência para análise de descrição de serviço
    Mantém compatibilidade com versão anterior
    
    Args:
        text: Texto a ser analisado
        **kwargs: Argumentos adicionais
        
    Returns:
        Resultado da análise
    """
    return enhanced_analyzer.analyze_text(text, **kwargs)


def get_enhanced_ml_info() -> Dict[str, Any]:
    """
    Retorna informações sobre o ML aprimorado
    
    Returns:
        Informações do sistema ML
    """
    return enhanced_analyzer.get_model_info()


# Função para teste
def test_enhanced_ml():
    """Testa o ML aprimorado com exemplos"""
    test_cases = [
        "Configuração completa do módulo NFe incluindo certificado digital",
        "Treinamento dos usuários sobre emissão de notas fiscais",
        "Suporte para resolver erro na transmissão de NFCe",
        "Implantação do sistema ERP na filial de São Paulo",
        "Análise de relatórios gerenciais e indicadores"
    ]
    
    print("🧪 Testando ML Aprimorado:")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        result = enhanced_analyzer.analyze_text(text)
        print(f"\n{i}. Texto: {text}")
        print(f"   Tipo: {result['tipo_servico']} ({result['confianca']:.1%})")
        print(f"   Palavras-chave: {', '.join(result['palavras_chave'])}")
        print(f"   Modelo: {result['modelo_usado']}")


if __name__ == "__main__":
    test_enhanced_ml()

