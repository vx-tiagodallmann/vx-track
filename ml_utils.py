"""
Módulo de utilitários de Machine Learning para o Assistente de Anotações (Versão Melhorada)
"""

import re
import json
from typing import List, Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import os

class TextProcessor:
    """Classe para processamento e análise de texto das anotações (Versão Melhorada)"""
    
    def __init__(self):
        self.service_classifier = None
        self.keywords_extractor = None
        self.setup_models()
    
    def setup_models(self):
        """Configura os modelos de ML"""
        # Pipeline para classificação de tipo de serviço
        self.service_classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words=None)),
            ('classifier', MultinomialNB())
        ])
        
        # Configurar extrator de palavras-chave com stop words em português
        portuguese_stop_words = [
            'de', 'da', 'do', 'das', 'dos', 'a', 'o', 'as', 'os', 'e', 'em', 'para', 'com', 'por',
            'na', 'no', 'nas', 'nos', 'um', 'uma', 'uns', 'umas', 'que', 'se', 'foi', 'foram',
            'ser', 'estar', 'ter', 'haver', 'ao', 'aos', 'à', 'às', 'pelo', 'pela', 'pelos', 'pelas'
        ]
        
        self.keywords_extractor = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 3),
            stop_words=portuguese_stop_words,
            min_df=1
        )
    
    def preprocess_text(self, text: str) -> str:
        """Pré-processa o texto removendo caracteres especiais e normalizando"""
        if not text:
            return ""
        
        # Converter para minúsculas
        text = text.lower()
        
        # Remover caracteres especiais, manter apenas letras, números e espaços
        text = re.sub(r'[^a-záàâãéèêíìîóòôõúùûç\s]', ' ', text)
        
        # Remover espaços múltiplos
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extrai palavras-chave do texto usando TF-IDF melhorado"""
        processed_text = self.preprocess_text(text)
        
        if not processed_text or len(processed_text.split()) < 3:
            return []
        
        try:
            # Fit e transform do texto
            tfidf_matrix = self.keywords_extractor.fit_transform([processed_text])
            
            # Obter nomes das features (palavras)
            feature_names = self.keywords_extractor.get_feature_names_out()
            
            # Obter scores TF-IDF
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Criar lista de (palavra, score) e ordenar por score
            word_scores = list(zip(feature_names, tfidf_scores))
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Filtrar palavras relevantes e retornar top N
            relevant_words = []
            for word, score in word_scores:
                if score > 0 and len(word) > 2:  # Palavras com mais de 2 caracteres
                    relevant_words.append(word)
                    if len(relevant_words) >= top_n:
                        break
            
            return relevant_words
        
        except Exception as e:
            print(f"Erro ao extrair palavras-chave: {e}")
            # Fallback: extrair palavras importantes manualmente
            important_words = []
            words = processed_text.split()
            
            # Palavras técnicas importantes
            tech_words = [
                'sistema', 'construshow', 'cliente', 'configuração', 'módulo', 'nfe', 'nfce',
                'fiscal', 'contábil', 'orion', 'multiadquirente', 'treinamento', 'implantação',
                'suporte', 'personalização', 'relatório', 'integração', 'estoque'
            ]
            
            for word in words:
                if word in tech_words and word not in important_words:
                    important_words.append(word)
                    if len(important_words) >= top_n:
                        break
            
            return important_words
    
    def classify_service_type(self, text: str) -> str:
        """Classifica o tipo de serviço baseado no texto da descrição (Melhorado)"""
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return "Outros"
        
        # Palavras-chave para cada tipo de serviço (melhoradas)
        service_keywords = {
            "Implantação": [
                "implantação", "implementação", "instalação", "setup", "configuração inicial",
                "primeiro acesso", "criação", "cadastro inicial", "parametrização inicial",
                "instalação do sistema", "primeira configuração", "setup inicial"
            ],
            "Configuração": [
                "configuração", "parametrização", "ajuste", "personalização", "customização",
                "configurar", "ajustar", "personalizar", "layout", "relatório", "parâmetros",
                "configuração de", "ajuste de", "parametrização de"
            ],
            "Treinamento": [
                "treinamento", "capacitação", "orientação", "explicação", "demonstração",
                "ensino", "tutorial", "apresentação", "instrução", "treinar", "capacitar",
                "orientar", "explicar", "demonstrar", "ensinar"
            ],
            "Suporte": [
                "suporte", "auxílio", "ajuda", "resolução", "problema", "erro", "dúvida",
                "dificuldade", "falha", "correção", "resolver", "solucionar", "corrigir",
                "resolver problema", "solução de", "correção de"
            ],
            "Personalização": [
                "personalização", "customização", "desenvolvimento", "criação de relatório",
                "layout personalizado", "funcionalidade específica", "adaptação", "customizar",
                "desenvolver", "criar relatório", "relatório específico", "desenvolvimento de"
            ]
        }
        
        # Contar ocorrências de palavras-chave para cada tipo (com pesos)
        type_scores = {}
        
        for service_type, keywords in service_keywords.items():
            score = 0
            for keyword in keywords:
                # Contar ocorrências exatas
                exact_count = processed_text.count(keyword)
                
                # Peso maior para palavras-chave mais específicas
                weight = len(keyword.split())  # Frases têm peso maior que palavras únicas
                
                score += exact_count * weight
            
            type_scores[service_type] = score
        
        # Retornar o tipo com maior score, ou "Outros" se nenhum score > 0
        if max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        else:
            return "Outros"
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extrai entidades nomeadas do texto (módulos, clientes, etc.) - Melhorado"""
        processed_text = self.preprocess_text(text)
        
        entities = {
            "modulos": [],
            "sistemas": [],
            "processos": []
        }
        
        # Padrões para módulos conhecidos (melhorados)
        modulo_patterns = [
            r"construshow", r"nfe", r"nfce", r"fiscal", r"contábil", r"contabil",
            r"orion", r"multiadquirente", r"hok", r"hók", r"estoque", r"financeiro"
        ]
        
        # Padrões para sistemas/processos
        sistema_patterns = [
            r"sistema", r"erp", r"software", r"aplicação", r"aplicacao",
            r"plataforma", r"ferramenta", r"módulo", r"modulo"
        ]
        
        processo_patterns = [
            r"emissão", r"emissao", r"transmissão", r"transmissao", r"importação", r"importacao",
            r"exportação", r"exportacao", r"integração", r"integracao", r"sincronização", r"sincronizacao",
            r"configuração", r"configuracao", r"parametrização", r"parametrizacao"
        ]
        
        # Buscar módulos
        for pattern in modulo_patterns:
            matches = re.findall(pattern, processed_text)
            for match in matches:
                if match.upper() not in entities["modulos"]:
                    entities["modulos"].append(match.upper())
        
        # Buscar sistemas
        for pattern in sistema_patterns:
            matches = re.findall(pattern, processed_text)
            for match in matches:
                if match.upper() not in entities["sistemas"]:
                    entities["sistemas"].append(match.upper())
        
        # Buscar processos
        for pattern in processo_patterns:
            matches = re.findall(pattern, processed_text)
            for match in matches:
                if match.upper() not in entities["processos"]:
                    entities["processos"].append(match.upper())
        
        return entities
    
    def summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """Cria um resumo do texto usando técnicas melhoradas de sumarização"""
        if not text or len(text.strip()) < 50:
            return text
        
        # Dividir em sentenças
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Calcular scores das sentenças baseado em palavras-chave importantes (melhorado)
        important_words = [
            "configuração", "implantação", "treinamento", "problema", "resolução",
            "cliente", "sistema", "módulo", "funcionalidade", "processo", "construshow",
            "nfe", "nfce", "fiscal", "orion", "multiadquirente", "sucesso", "satisfação"
        ]
        
        sentence_scores = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Score baseado em palavras importantes
            for word in important_words:
                if word in sentence_lower:
                    score += 2  # Peso maior para palavras importantes
            
            # Score baseado na posição (primeiras e últimas sentenças são importantes)
            position_score = 0
            sentence_index = sentences.index(sentence)
            if sentence_index == 0:  # Primeira sentença
                position_score = 1.5
            elif sentence_index == len(sentences) - 1:  # Última sentença
                position_score = 1.2
            elif sentence_index < len(sentences) * 0.3:  # Primeiro terço
                position_score = 1.1
            
            # Score baseado no tamanho (sentenças muito curtas ou muito longas são penalizadas)
            length_score = 1.0
            if len(sentence) < 30:
                length_score = 0.5
            elif len(sentence) > 200:
                length_score = 0.8
            elif 50 <= len(sentence) <= 150:
                length_score = 1.2  # Tamanho ideal
            
            final_score = (score + position_score) * length_score
            sentence_scores.append((sentence, final_score))
        
        # Ordenar por score e pegar as melhores sentenças
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Pegar as top sentenças e reordenar pela posição original
        top_sentences = [s[0] for s in sentence_scores[:max_sentences]]
        
        # Reordenar pela posição original no texto
        summary_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                summary_sentences.append(sentence)
        
        return ". ".join(summary_sentences) + "."
    
    def analyze_annotation(self, text: str) -> Dict:
        """Análise completa de uma anotação (Melhorada)"""
        if not text:
            return {
                "tipo_servico_sugerido": "Outros",
                "palavras_chave": [],
                "entidades": {"modulos": [], "sistemas": [], "processos": []},
                "resumo": "",
                "confianca": 0.0
            }
        
        # Realizar todas as análises
        tipo_servico = self.classify_service_type(text)
        palavras_chave = self.extract_keywords(text)
        entidades = self.extract_entities(text)
        resumo = self.summarize_text(text)
        
        # Calcular confiança melhorada
        confianca = self.calculate_confidence(text, tipo_servico, palavras_chave, entidades)
        
        return {
            "tipo_servico_sugerido": tipo_servico,
            "palavras_chave": palavras_chave,
            "entidades": entidades,
            "resumo": resumo,
            "confianca": confianca
        }
    
    def calculate_confidence(self, text: str, tipo_servico: str, palavras_chave: List[str], entidades: Dict) -> float:
        """Calcula a confiança da análise baseada em múltiplos fatores"""
        confidence_factors = []
        
        # Fator 1: Tamanho do texto (textos muito curtos têm baixa confiança)
        text_length = len(text.strip())
        if text_length < 50:
            length_confidence = 0.3
        elif text_length < 100:
            length_confidence = 0.6
        elif text_length < 200:
            length_confidence = 0.8
        else:
            length_confidence = 1.0
        confidence_factors.append(length_confidence)
        
        # Fator 2: Número de palavras-chave encontradas
        keyword_confidence = min(len(palavras_chave) / 5.0, 1.0)  # Máximo 5 palavras-chave
        confidence_factors.append(keyword_confidence)
        
        # Fator 3: Presença de entidades técnicas
        total_entities = sum(len(entities) for entities in entidades.values())
        entity_confidence = min(total_entities / 3.0, 1.0)  # Máximo 3 entidades
        confidence_factors.append(entity_confidence)
        
        # Fator 4: Confiança na classificação do tipo de serviço
        if tipo_servico != "Outros":
            type_confidence = 0.8
        else:
            type_confidence = 0.4
        confidence_factors.append(type_confidence)
        
        # Calcular confiança média ponderada
        weights = [0.2, 0.3, 0.2, 0.3]  # Pesos para cada fator
        weighted_confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
        
        return round(weighted_confidence, 2)

# Instância global do processador melhorado
text_processor = TextProcessor()



class MLAnalyzer:
    """Analisador de ML para classificação e extração de informações de texto"""
    
    def __init__(self):
        """Inicializa o analisador ML usando o TextProcessor existente"""
        self.processor = text_processor
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analisa um texto e retorna classificação e informações extraídas
        
        Args:
            text: Texto a ser analisado
            
        Returns:
            Dicionário com análise completa
        """
        if not text or not text.strip():
            return {
                "tipo_servico": "Não identificado",
                "confianca": 0.0,
                "palavras_chave": [],
                "resumo": "",
                "modulos_identificados": []
            }
        
        # Usa o processador existente para análise completa
        resultado = self.processor.analyze_annotation(text)
        
        # Adapta o formato de saída para compatibilidade
        return {
            "tipo_servico": resultado.get("tipo_servico_sugerido", "Outros"),
            "confianca": resultado.get("confianca", 0.0),
            "palavras_chave": resultado.get("palavras_chave", []),
            "resumo": resultado.get("resumo", ""),
            "modulos_identificados": resultado.get("entidades", {}).get("modulos", [])
        }

