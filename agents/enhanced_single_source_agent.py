from typing import List, Dict, Any
from PIL import Image
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline
import vllm
import re

# Constants for vLLM and model configuration (optimized for a single NVIDIA L40s GPU)
VLLM_TENSOR_PARALLEL_SIZE = 1        # no tensor parallelism (single GPU)
VLLM_GPU_MEMORY_UTILIZATION = 0.85   # use 85% of GPU memory to leave headroom
MAX_MODEL_LEN = 8192                 # maximum context length for the model
MAX_NUM_SEQS = 2                     # max parallel sequences per generation call
MAX_GENERATION_TOKENS = 75          # max tokens to generate for answers
MAX_KNOWLEDGE_TOKENS = 600          # increased from 400 for more context

class EnhancedSingleSourceAgent(BaseAgent):
    """
     Single Source Agent for CRAG-MM Task 1.
    
    Improvements over the original:
    1. Better image description generation with more targeted prompts
    2. Improved entity attribute cleaning with more comprehensive patterns
    3.  knowledge selection and ranking
    4. Better prompt engineering for final answer generation
    5. More robust fallback strategies
    """
 
    def __init__(self, search_pipeline: UnifiedSearchPipeline, 
                 model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", 
                 max_gen_len: int = 64):

        super().__init__(search_pipeline)
        self.model_name = model_name
        self.max_gen_len = max_gen_len
        
        # Initialize the model using vLLM
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1}  # allow only 1 image per prompt
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("Enhanced Single Source Agent initialized successfully.")
    
    def get_batch_size(self) -> int:
        return 8
    
    def batch_analyze_images(self, images: List[Image.Image], queries: List[str]) -> List[str]:
        """
        Generate detailed image analyses tailored to the specific queries.
        This replaces the generic image summarization with query-aware analysis.
        """
        analyses = []
        formatted_prompts = []
        
        for img, query in zip(images, queries):
            # Create a more targeted system prompt based on query type
            system_prompt = self._get_analysis_system_prompt(query)
            
            # Include the query in the analysis request
            analysis_request = f"Given this question: '{query}', please describe what you see in the image with focus on details that might help answer the question. Include any visible text, brands, objects, landmarks, or other identifying features."

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "image"}]},
                {"role": "user", "content": analysis_request}
            ]
            
            prompt_str = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            formatted_prompts.append(prompt_str)

        inputs = [
            {"prompt": prompt, "multi_modal_data": {"image": img}}
            for prompt, img in zip(formatted_prompts, images)
        ]
        
        # Use vLLM to generate analyses for all images in batch
        desc_outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=128,  # increased for more detailed analysis
                skip_special_tokens=True
            )
        )
        
        # Extract the generated text for each image analysis
        for output in desc_outputs:
            desc_text = output.outputs[0].text.strip()
            analyses.append(desc_text if desc_text else "")
        
        return analyses
    
    def _get_analysis_system_prompt(self, query: str) -> str:
        """
        Generate a targeted system prompt based on the query type.
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['brand', 'company', 'manufacturer', 'maker']):
            return "You are an expert at identifying brands, logos, and company information from images. Focus on any visible brand names, logos, or identifying marks."
        elif any(word in query_lower for word in ['cost', 'price', 'buy', 'purchase', 'amazon', 'walmart']):
            return "You are an expert at identifying products and their commercial details. Focus on product names, models, and any visible packaging or labeling."
        elif any(word in query_lower for word in ['material', 'made', 'construction', 'fabric']):
            return "You are an expert at identifying materials and construction details. Focus on textures, materials, and manufacturing details visible in the image."
        elif any(word in query_lower for word in ['color', 'colours', 'options', 'variants']):
            return "You are an expert at identifying colors, variations, and options. Focus on color details and any visible variety indicators."
        elif any(word in query_lower for word in ['size', 'capacity', 'volume', 'dimensions']):
            return "You are an expert at identifying size and capacity information. Focus on any visible measurements, size indicators, or capacity markings."
        else:
            return "You are an expert at analyzing images and identifying key details. Focus on all visible elements that could help answer questions about the image."
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Hard-trim a string to at most `max_tokens` tokenizer tokens."""
        if not text:
            return ""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_tokens:
            return text
        # decode only the first max_tokens ids back to text
        return self.tokenizer.decode(ids[:max_tokens], skip_special_tokens=True) + "..."
    
    def _clean_entity_attributes(self, attrs: Dict[str, str]) -> str:
        """
        Enhanced entity attribute cleaning with more comprehensive patterns.
        """
        # Fields to ignore (not useful for answering questions)
        ignore_keys = {
            "image", "image_size", "coordinates", "mapframe_wikidata", 
            "website", "commons", "wikidata", "geonames"
        }
        
        facts = []
        for key, value in attrs.items():
            if key in ignore_keys or not value:
                continue
            
            # Clean the value more thoroughly
            cleaned_val = self._clean_attribute_value(value)
            if not cleaned_val.strip():
                continue

            # Format the key nicely
            pretty_key = self._format_attribute_key(key)
            facts.append(f"{pretty_key}: {cleaned_val}")
        
        knowledge_text = "; ".join(facts)
        return knowledge_text
    
    def _clean_attribute_value(self, value: str) -> str:
        """
        Comprehensive cleaning of attribute values.
        """
        cleaned_val = str(value)
        
        # Replace HTML line breaks
        cleaned_val = re.sub(r'<br\s*/?>', ', ', cleaned_val)
        
        # Remove Wiki link markup [[text|display]] or [[text]]
        def replace_wiki_links(match):
            link_text = match.group(1)
            if '|' in link_text:
                return link_text.split('|')[-1]  # Keep display text
            return link_text
        
        cleaned_val = re.sub(r'\[\[([^\]]+)\]\]', replace_wiki_links, cleaned_val)
        
        # Handle convert templates more robustly
        def replace_convert_template(match):
            template_content = match.group(1)
            parts = template_content.split('|')
            if len(parts) >= 3 and parts[0].strip().lower() == 'convert':
                try:
                    number = parts[1].strip()
                    unit = parts[2].strip()
                    return f"{number} {unit}"
                except:
                    return ""
            return ""
        
        cleaned_val = re.sub(r'\{\{([^}]+)\}\}', replace_convert_template, cleaned_val)
        
        # Remove other template patterns
        cleaned_val = re.sub(r'\{\{[^}]*\}\}', '', cleaned_val)
        
        # Clean up URLs and references
        cleaned_val = re.sub(r'https?://[^\s]+', '', cleaned_val)
        
        # Remove citation patterns
        cleaned_val = re.sub(r'\[\d+\]', '', cleaned_val)
        
        # Clean up whitespace and punctuation
        cleaned_val = re.sub(r'\s+', ' ', cleaned_val)
        cleaned_val = cleaned_val.strip().strip(',').strip(';')
        
        return cleaned_val
    
    def _format_attribute_key(self, key: str) -> str:
        """
        Format attribute keys for better readability.
        """
        # Handle common abbreviations and special cases
        key_mappings = {
            'ans_full': 'Answer',
            'start_date': 'Started',
            'completion_date': 'Completed',
            'floor_count': 'Floors',
            'floor_area': 'Area',
            'architectural_style': 'Style',
            'building_type': 'Type',
            'main_contractor': 'Contractor',
            'structural_engineer': 'Engineer',
        }
        
        if key in key_mappings:
            return key_mappings[key]
        
        # Convert snake_case to Title Case
        formatted = key.replace('_', ' ').title()
        
        # Handle some common words that should be capitalized differently
        replacements = {
            'Url': 'URL',
            'Id': 'ID',
            'Gps': 'GPS',
            'Usa': 'USA',
            'Uk': 'UK',
        }
        
        for old, new in replacements.items():
            formatted = formatted.replace(old, new)
        
        return formatted
    
    def _select_best_knowledge(self, search_results: List[Dict], query: str) -> str:
        """
        Select and format the best knowledge from search results based on the query.
        """
        if not search_results:
            return ""
        
        # Score entities based on relevance to query
        scored_entities = []
        query_lower = query.lower()
        
        for result in search_results:
            entities = result.get('entities', [])
            for entity in entities:
                attrs = entity.get('entity_attributes', {})
                if not attrs:
                    continue
                
                # Calculate relevance score
                score = self._calculate_entity_relevance(attrs, query_lower)
                if score > 0:
                    scored_entities.append((score, attrs))
        
        if not scored_entities:
            # Fallback to first available entity
            for result in search_results:
                entities = result.get('entities', [])
                if entities and entities[0].get('entity_attributes'):
                    return self._clean_entity_attributes(entities[0]['entity_attributes'])
            return ""
        
        # Sort by relevance score and take the best
        scored_entities.sort(key=lambda x: x[0], reverse=True)
        best_attrs = scored_entities[0][1]
        
        return self._clean_entity_attributes(best_attrs)
    
    def _calculate_entity_relevance(self, attrs: Dict[str, str], query_lower: str) -> float:
        """
        Calculate how relevant an entity's attributes are to the query.
        """
        score = 0.0
        
        # Keywords that indicate different types of queries
        query_keywords = {
            'brand': ['brand', 'company', 'manufacturer', 'maker', 'owns'],
            'price': ['cost', 'price', 'buy', 'purchase', 'amazon', 'walmart', 'expensive'],
            'material': ['material', 'made', 'construction', 'fabric', 'steel', 'wood', 'plastic'],
            'size': ['size', 'capacity', 'volume', 'dimensions', 'big', 'small', 'hold'],
            'color': ['color', 'colours', 'options', 'variants', 'available'],
            'location': ['where', 'location', 'address', 'place', 'city', 'country'],
            'time': ['when', 'date', 'year', 'old', 'built', 'made', 'started', 'completed'],
        }
        
        # Check which category the query belongs to
        query_category = None
        for category, keywords in query_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                query_category = category
                break
        
        # Score based on relevant attributes
        for key, value in attrs.items():
            key_lower = key.lower()
            value_lower = str(value).lower()
            
            # Base score for having any information
            if value and str(value).strip():
                score += 0.1
            
            # Category-specific scoring
            if query_category == 'brand' and any(word in key_lower for word in ['name', 'brand', 'company', 'manufacturer', 'owner']):
                score += 2.0
            elif query_category == 'price' and any(word in key_lower for word in ['price', 'cost', 'value']):
                score += 2.0
            elif query_category == 'material' and any(word in key_lower for word in ['material', 'construction', 'made']):
                score += 2.0
            elif query_category == 'size' and any(word in key_lower for word in ['size', 'capacity', 'volume', 'area', 'floor']):
                score += 2.0
            elif query_category == 'color' and 'color' in key_lower:
                score += 2.0
            elif query_category == 'location' and any(word in key_lower for word in ['address', 'location', 'city', 'country']):
                score += 2.0
            elif query_category == 'time' and any(word in key_lower for word in ['date', 'year', 'start', 'completion', 'opening']):
                score += 2.0
            
            # Boost score if query terms appear in the value
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3 and word in value_lower:
                    score += 0.5
        
        return score
    
    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]]
    ) -> List[str]:

        batch_size = len(queries)
        responses: List[str] = [None] * batch_size
        
        # Step 1: Generate query-aware image analyses
        image_analyses = self.batch_analyze_images(images, queries)
        
        # Step 2: Perform image search and extract knowledge
        knowledge_list: List[str] = [None] * batch_size
        for i, (query, img, analysis) in enumerate(zip(queries, images, image_analyses)):
            try:
                results = self.search_pipeline(img, k=5)  # Get more results for better selection
                if not results or len(results) == 0:
                    responses[i] = "I don't know."
                    continue
                
                # Use enhanced knowledge selection
                knowledge_text = self._select_best_knowledge(results, query)
                if not knowledge_text.strip():
                    responses[i] = "I don't know."
                    continue
                
                # Truncate knowledge to fit context
                knowledge_text = self._truncate_to_tokens(knowledge_text, MAX_KNOWLEDGE_TOKENS)
                knowledge_list[i] = knowledge_text
                
            except Exception as e:
                print(f"Error in search for query {i}: {e}")
                responses[i] = "I don't know."
                continue
        
        # Step 3: Generate answers using enhanced prompting
        gen_inputs = []
        idx_map = []
        
        for i, (query, knowledge, analysis) in enumerate(zip(queries, knowledge_list, image_analyses)):
            if responses[i] is not None:
                continue

            # Create enhanced system prompt with image analysis
            system_prompt = self._create_enhanced_system_prompt(knowledge, analysis, query)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "image"}]},
                {"role": "user", "content": query}
            ]
            
            prompt_str = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            
            gen_inputs.append({
                "prompt": prompt_str,
                "multi_modal_data": {"image": images[i]}
            })
            idx_map.append(i)
        
        # Generate answers in batch
        if gen_inputs:
            outputs = self.llm.generate(
                gen_inputs,
                sampling_params=vllm.SamplingParams(
                    temperature=0,
                    top_p=1,
                    max_tokens=self.max_gen_len,
                    skip_special_tokens=True
                )
            )
            
            for output, orig_idx in zip(outputs, idx_map):
                answer = output.outputs[0].text.strip()
                responses[orig_idx] = answer if answer else "I don't know."
        
        # Fill any remaining None responses
        for i in range(batch_size):
            if responses[i] is None:
                responses[i] = "I don't know."
        
        return responses
    
    def _create_enhanced_system_prompt(self, knowledge: str, analysis: str, query: str) -> str:
        """
        Create an enhanced system prompt that incorporates knowledge, image analysis, and query context.
        """
        base_prompt = (
            "You are a helpful assistant that answers questions about images using provided information. "
            "You have access to factual information about the image and a detailed visual analysis."
        )
        
        if knowledge:
            base_prompt += f"\n\nFactual Information:\n{knowledge}"
        
        if analysis:
            base_prompt += f"\n\nVisual Analysis:\n{analysis}"
        
        base_prompt += (
            "\n\nInstructions:\n"
            "1. Answer the question directly and concisely using the provided information\n"
            "2. If the factual information doesn't contain the answer, use the visual analysis\n"
            "3. If neither source has sufficient information, respond with 'I don't know.'\n"
            "4. Do not make up or guess information not present in the sources\n"
            "5. Be specific and factual in your response"
        )
        
        return base_prompt