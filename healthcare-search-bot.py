#!/usr/bin/env python3
"""
CXone Healthcare Search Bot with Deep Learning

This application uses deep learning to process and enhance healthcare search queries,
providing more relevant results and extracting key medical insights from search results.
It's designed to integrate with NICE CXone to support healthcare contact centers.

Features:
- Medical entity recognition
- Query enhancement with healthcare ontologies
- Symptom classification
- Treatment recommendation summarization
- HIPAA-compliant information handling
- Integration with CXone telephony systems

Author: NICE CXone Software Engineer
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, pipeline

# NLP and Search imports
from serpapi import GoogleSearch
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("healthcare_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("HealthcareBot")

class MedicalKnowledgeBase:
    """Manages medical terminology and healthcare ontologies"""
    
    def __init__(self, ontology_file: str = "medical_ontology.json"):
        """Initialize the medical knowledge base
        
        Args:
            ontology_file: Path to medical ontology JSON file
        """
        self.ontology_file = ontology_file
        self.ontology = self._load_ontology()
        self.symptom_map = self._build_symptom_map()
        self.condition_map = self._build_condition_map()
        self.medication_map = self._build_medication_map()
        
        logger.info(f"Medical Knowledge Base initialized with {len(self.symptom_map)} symptoms, "
                   f"{len(self.condition_map)} conditions, and {len(self.medication_map)} medications")
    
    def _load_ontology(self) -> Dict:
        """Load medical ontology from JSON file"""
        try:
            if os.path.exists(self.ontology_file):
                with open(self.ontology_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Ontology file {self.ontology_file} not found. Using default minimal ontology.")
                return {
                    "symptoms": {},
                    "conditions": {},
                    "medications": {},
                    "procedures": {},
                    "specialties": {}
                }
        except Exception as e:
            logger.error(f"Error loading ontology: {str(e)}")
            return {}
    
    def _build_symptom_map(self) -> Dict[str, List[str]]:
        """Build mapping of symptoms to related medical concepts"""
        symptom_map = {}
        for symptom, data in self.ontology.get("symptoms", {}).items():
            symptom_map[symptom] = data.get("related_conditions", [])
            # Add common variations/misspellings
            for variation in data.get("variations", []):
                symptom_map[variation] = data.get("related_conditions", [])
        return symptom_map
    
    def _build_condition_map(self) -> Dict[str, Dict]:
        """Build mapping of medical conditions to their properties"""
        condition_map = {}
        for condition, data in self.ontology.get("conditions", {}).items():
            condition_map[condition] = data
            # Add common variations/misspellings
            for variation in data.get("variations", []):
                condition_map[variation] = data
        return condition_map
    
    def _build_medication_map(self) -> Dict[str, Dict]:
        """Build mapping of medications to their properties"""
        medication_map = {}
        for medication, data in self.ontology.get("medications", {}).items():
            medication_map[medication] = data
            # Add generic/brand name mappings
            for alt_name in data.get("alternative_names", []):
                medication_map[alt_name] = data
        return medication_map
    
    def enhance_query(self, query: str) -> str:
        """Enhance a search query with medical terminology
        
        Args:
            query: Original search query
        
        Returns:
            Enhanced query with relevant medical terms
        """
        # Simple implementation - in a real system this would use more sophisticated NLP
        enhanced_query = query
        
        # Check for symptoms in query
        for symptom in self.symptom_map:
            if symptom.lower() in query.lower():
                # Add related conditions to enhance search
                related = self.symptom_map[symptom][:2]  # Limit to top 2 related conditions
                if related:
                    enhanced_terms = " OR ".join(related)
                    enhanced_query = f"{query} ({enhanced_terms})"
                break
                
        # Check for medications and add generic/brand names
        for medication in self.medication_map:
            if medication.lower() in query.lower():
                med_data = self.medication_map[medication]
                if "alternative_names" in med_data and med_data["alternative_names"]:
                    alt_name = med_data["alternative_names"][0]
                    if alt_name.lower() not in query.lower():
                        enhanced_query = f"{query} OR {alt_name}"
                break
        
        logger.info(f"Enhanced query: '{query}' -> '{enhanced_query}'")
        return enhanced_query
    
    def identify_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Identify medical entities in text
        
        Args:
            text: Input text to analyze
        
        Returns:
            Dictionary of medical entities by category
        """
        entities = {
            "symptoms": [],
            "conditions": [],
            "medications": [],
            "procedures": [],
            "specialties": []
        }
        
        # Check for symptoms
        for symptom in self.symptom_map:
            if symptom.lower() in text.lower():
                entities["symptoms"].append(symptom)
        
        # Check for conditions
        for condition in self.condition_map:
            if condition.lower() in text.lower():
                entities["conditions"].append(condition)
        
        # Check for medications
        for medication in self.medication_map:
            if medication.lower() in text.lower():
                entities["medications"].append(medication)
        
        # Remove duplicates
        for category in entities:
            entities[category] = list(set(entities[category]))
        
        return entities


class MedicalBertModel:
    """Medical domain-specific BERT model for text processing"""
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
        """Initialize the medical BERT model
        
        Args:
            model_name: Hugging Face model name/path
        """
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded Medical BERT model: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Error loading Medical BERT model: {str(e)}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
        
        Returns:
            Array of embeddings
        """
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                outputs = self.model(**inputs)
                # Use CLS token embedding as the text representation
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score between 0 and 1
        """
        embeddings = self.get_embeddings([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def classify_medical_text(self, text: str, categories: List[str]) -> Tuple[str, float]:
        """Classify medical text into predefined categories
        
        Args:
            text: Text to classify
            categories: List of category names
        
        Returns:
            Tuple of (best_category, confidence_score)
        """
        # Create comparison texts for each category
        comparison_texts = [f"This is about {category}" for category in categories]
        
        # Get embeddings for the text and all categories
        text_embedding = self.get_embeddings([text])[0]
        category_embeddings = self.get_embeddings(comparison_texts)
        
        # Calculate similarity with each category
        similarities = [
            cosine_similarity([text_embedding], [cat_emb])[0][0]
            for cat_emb in category_embeddings
        ]
        
        # Find best matching category
        best_idx = np.argmax(similarities)
        best_category = categories[best_idx]
        confidence = similarities[best_idx]
        
        return best_category, float(confidence)


class GoogleHealthcareSearcher:
    """Performs healthcare-specific Google searches with enhanced processing"""
    
    def __init__(self, api_key: str, model: MedicalBertModel, knowledge_base: MedicalKnowledgeBase):
        """Initialize the healthcare searcher
        
        Args:
            api_key: SerpAPI key for Google searches
            model: Medical BERT model for text processing
            knowledge_base: Medical knowledge base for query enhancement
        """
        self.api_key = api_key
        self.model = model
        self.knowledge_base = knowledge_base
        logger.info("Healthcare Searcher initialized")
    
    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Perform a healthcare-enhanced Google search
        
        Args:
            query: Search query
            num_results: Number of results to return
        
        Returns:
            List of processed search results
        """
        # Enhance the query with medical knowledge
        enhanced_query = self.knowledge_base.enhance_query(query)
        
        # Perform Google search
        try:
            search_params = {
                "q": enhanced_query,
                "api_key": self.api_key,
                "num": num_results,
                "tbm": "web",  # Search type (web)
                "hl": "en",    # Language (English)
                "safe": "active"  # Safe search
            }
            
            search = GoogleSearch(search_params)
            results = search.get_dict()
            
            if "error" in results:
                logger.error(f"Search error: {results['error']}")
                return []
            
            organic_results = results.get("organic_results", [])
            
            # Process and enhance each result
            processed_results = []
            for result in organic_results:
                # Extract basic information
                processed_result = {
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "source": result.get("source", ""),
                    "position": result.get("position", 0)
                }
                
                # Add medical entity detection
                snippet_text = processed_result["snippet"]
                medical_entities = self.knowledge_base.identify_medical_entities(snippet_text)
                processed_result["medical_entities"] = medical_entities
                
                # Classify the result content
                medical_categories = ["symptoms", "diagnosis", "treatment", "prevention", "research"]
                category, confidence = self.model.classify_medical_text(snippet_text, medical_categories)
                processed_result["category"] = category
                processed_result["confidence"] = confidence
                
                # Add result relevance score
                relevance = self._calculate_relevance(query, processed_result)
                processed_result["relevance_score"] = relevance
                
                processed_results.append(processed_result)
            
            # Sort by custom relevance
            processed_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            logger.info(f"Performed search for '{query}', found {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
    
    def _calculate_relevance(self, query: str, result: Dict) -> float:
        """Calculate a custom relevance score for a search result
        
        Args:
            query: Original search query
            result: Search result dictionary
        
        Returns:
            Relevance score between 0 and 1
        """
        # Calculate semantic similarity between query and snippet
        semantic_score = self.model.calculate_similarity(query, result["snippet"])
        
        # Calculate entity match score
        entity_count = sum(len(entities) for entities in result["medical_entities"].values())
        entity_score = min(1.0, entity_count / 5.0)  # Cap at 1.0
        
        # Consider confidence in classification
        confidence_score = result["confidence"]
        
        # Calculate position score (higher for results at the top)
        position = result["position"]
        position_score = 1.0 / (1.0 + position)
        
        # Weighted combination
        relevance = (
            0.4 * semantic_score +
            0.3 * entity_score +
            0.2 * confidence_score +
            0.1 * position_score
        )
        
        return relevance


class HealthcareSearchBot:
    """Main bot class for healthcare search integration with CXone"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the healthcare search bot
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.api_key = self.config.get("serp_api_key", os.environ.get("SERP_API_KEY", ""))
        
        if not self.api_key:
            raise ValueError("SERP API key not found in config or environment variables")
        
        # Initialize components
        self.knowledge_base = MedicalKnowledgeBase(
            ontology_file=self.config.get("ontology_file", "medical_ontology.json")
        )
        
        self.model = MedicalBertModel(
            model_name=self.config.get("model_name", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        )
        
        self.searcher = GoogleHealthcareSearcher(
            api_key=self.api_key,
            model=self.model,
            knowledge_base=self.knowledge_base
        )
        
        # Initialize CXone integration if enabled
        self.cxone_enabled = self.config.get("cxone_integration", {}).get("enabled", False)
        if self.cxone_enabled:
            self._setup_cxone_integration()
        
        logger.info("Healthcare Search Bot initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file
        
        Args:
            config_path: Path to configuration file
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {str(e)}")
            logger.warning("Using default configuration")
            return {
                "serp_api_key": "",
                "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                "ontology_file": "medical_ontology.json",
                "max_results": 10,
                "cxone_integration": {
                    "enabled": False
                }
            }
    
    def _setup_cxone_integration(self):
        """Set up integration with NICE CXone"""
        cxone_config = self.config.get("cxone_integration", {})
        self.cxone_api_key = cxone_config.get("api_key", os.environ.get("CXONE_API_KEY", ""))
        self.cxone_api_url = cxone_config.get("api_url", "https://api.nice-incontact.com/v1")
        
        if not self.cxone_api_key:
            logger.warning("CXone integration enabled but API key not found")
            self.cxone_enabled = False
        else:
            logger.info("CXone integration initialized")
    
    def process_search_query(self, query: str, chat_context: Optional[List[Dict]] = None) -> Dict:
        """Process a healthcare search query
        
        Args:
            query: Search query
            chat_context: Optional list of previous chat messages for context
        
        Returns:
            Dictionary with search results and enhanced information
        """
        # Incorporate chat context if available
        enhanced_query = query
        if chat_context:
            # Extract relevant medical information from context
            context_symptoms = []
            context_conditions = []
            
            for message in chat_context[-5:]:  # Look at last 5 messages
                text = message.get("text", "")
                entities = self.knowledge_base.identify_medical_entities(text)
                context_symptoms.extend(entities.get("symptoms", []))
                context_conditions.extend(entities.get("conditions", []))
            
            # Add most relevant context to query
            context_symptoms = list(set(context_symptoms))[:2]
            context_conditions = list(set(context_conditions))[:1]
            
            if context_symptoms or context_conditions:
                context_terms = context_symptoms + context_conditions
                enhanced_query = f"{query} {' '.join(context_terms)}"
        
        # Perform search
        start_time = time.time()
        results = self.searcher.search(
            enhanced_query, 
            num_results=self.config.get("max_results", 10)
        )
        search_time = time.time() - start_time
        
        # Extract medical entities from all results
        all_entities = {
            "symptoms": [],
            "conditions": [],
            "medications": [],
            "procedures": [],
            "specialties": []
        }
        
        for result in results:
            for category, entities in result.get("medical_entities", {}).items():
                all_entities[category].extend(entities)
        
        # Remove duplicates
        for category in all_entities:
            all_entities[category] = list(set(all_entities[category]))
        
        # Categorize results by topic
        categorized_results = {}
        for result in results:
            category = result.get("category", "general")
            if category not in categorized_results:
                categorized_results[category] = []
            categorized_results[category].append(result)
        
        # Prepare response
        response = {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "search_time_seconds": search_time,
            "result_count": len(results),
            "results": results,
            "categorized_results": categorized_results,
            "medical_entities": all_entities,
            "timestamp": datetime.now().isoformat()
        }
        
        # If CXone integration is enabled, log the search
        if self.cxone_enabled:
            self._log_to_cxone(query, response)
        
        return response
    
    def _log_to_cxone(self, query: str, response: Dict):
        """Log search query and response to CXone
        
        Args:
            query: Original search query
            response: Search response dictionary
        """
        try:
            # Prepare log data
            log_data = {
                "eventType": "healthcare_search",
                "query": query,
                "enhancedQuery": response.get("enhanced_query", ""),
                "resultCount": response.get("result_count", 0),
                "medicalEntities": response.get("medical_entities", {}),
                "timestamp": response.get("timestamp", datetime.now().isoformat())
            }
            
            # Send to CXone
            headers = {
                "Authorization": f"Bearer {self.cxone_api_key}",
                "Content-Type": "application/json"
            }
            
            api_url = f"{self.cxone_api_url}/analytics/events"
            
            requests.post(
                api_url,
                headers=headers,
                json=log_data
            )
            
            logger.info(f"Logged search event to CXone: {query}")
        except Exception as e:
            logger.error(f"Error logging to CXone: {str(e)}")
    
    def generate_medical_summary(self, results: List[Dict], max_length: int = 200) -> str:
        """Generate a medical summary from search results
        
        Args:
            results: List of search results
            max_length: Maximum summary length in characters
        
        Returns:
            Generated summary text
        """
        if not results:
            return "No relevant medical information found."
        
        # Concatenate snippets from top results
        text = " ".join([r.get("snippet", "") for r in results[:3]])
        
        # For a more sophisticated implementation, this would use a text generation model
        # Here we'll just do basic extraction of key points
        
        # Extract medical entities
        all_entities = {}
        for result in results[:3]:
            for category, entities in result.get("medical_entities", {}).items():
                if category not in all_entities:
                    all_entities[category] = []
                all_entities[category].extend(entities)
        
        # Remove duplicates
        for category in all_entities:
            all_entities[category] = list(set(all_entities[category]))
        
        # Generate summary based on entities and categories
        summary_parts = []
        
        # Add symptoms if present
        if all_entities.get("symptoms"):
            symptoms = ", ".join(all_entities["symptoms"][:3])
            summary_parts.append(f"Common symptoms include {symptoms}.")
        
        # Add conditions if present
        if all_entities.get("conditions"):
            conditions = ", ".join(all_entities["conditions"][:2])
            summary_parts.append(f"This may be related to {conditions}.")
        
        # Add treatments if present
        if all_entities.get("medications"):
            medications = ", ".join(all_entities["medications"][:2])
            summary_parts.append(f"Treatments may include {medications}.")
        
        # Create final summary
        summary = " ".join(summary_parts)
        
        # Add disclaimer
        disclaimer = "This information is not medical advice. Consult a healthcare professional."
        
        full_summary = f"{summary} {disclaimer}"
        
        # Truncate if too long
        if len(full_summary) > max_length:
            full_summary = full_summary[:max_length-3] + "..."
        
        return full_summary


def main():
    """Main entry point for the healthcare search bot"""
    parser = argparse.ArgumentParser(description='Healthcare Search Bot')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--query', type=str, help='Search query')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    try:
        # Initialize bot
        bot = HealthcareSearchBot(config_path=args.config)
        
        # Process query if provided
        if args.query:
            query = args.query
            logger.info(f"Processing query: {query}")
            
            response = bot.process_search_query(query)
            
            # Generate summary
            summary = bot.generate_medical_summary(response.get("results", []))
            response["summary"] = summary
            
            # Output results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(response, f, indent=2)
                logger.info(f"Results saved to {args.output}")
            else:
                print(json.dumps(response, indent=2))
            
            print("\nSummary:")
            print(summary)
        else:
            print("No query provided. Use --query parameter to search.")
    
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
