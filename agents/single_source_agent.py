from typing import List, Dict, Any
from PIL import Image
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline
import vllm

# Constants for vLLM and model configuration (optimized for a single NVIDIA L40s GPU)
VLLM_TENSOR_PARALLEL_SIZE = 1        # no tensor parallelism (single GPU)
VLLM_GPU_MEMORY_UTILIZATION = 0.85   # use 85% of GPU memory to leave headroom
MAX_MODEL_LEN = 8192                 # maximum context length for the model
MAX_NUM_SEQS = 2                     # max parallel sequences per generation call
MAX_GENERATION_TOKENS = 75          # max tokens to generate for answers
MAX_KNOWLEDGE_TOKENS = 400

class SingleSourceAgent(BaseAgent):
 
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
            limit_mm_per_prompt={"image": 1}  # allow only 1 image per prompt (Task 1 has single image inputs)
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("Model and tokenizer initialized successfully.")
    
    def get_batch_size(self) -> int:
        return 8
    
    def batch_summarize_images(self, images: List[Image.Image]) -> List[str]:
        summaries = []

        formatted_prompts = []
        for img in images:
            system_prompt = (
                "You are a helpful assistant specialized in visual analysis. "
                "Describe the image in a single concise sentence mentioning the key objects or landmarks."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [ {"type": "image"} ]},
                {"role": "user", "content": "Please briefly describe the image."}
            ]
            
            prompt_str = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            formatted_prompts.append(prompt_str)

        inputs = [
            {"prompt": prompt, "multi_modal_data": {"image": img}}
            for prompt, img in zip(formatted_prompts, images)
        ]
        # Use vLLM to generate descriptions for all images in batch
        desc_outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=64,              # limit to a short description
                skip_special_tokens=True
            )
        )
        # Extract the generated text for each image description
        for output in desc_outputs:
            desc_text = output.outputs[0].text.strip()
            summaries.append(desc_text if desc_text else "")
        return summaries
    
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Hard-trim a string to at most `max_tokens` tokenizer tokens."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_tokens:
            return text
        # decode only the first max_tokens ids back to text
        return self.tokenizer.decode(ids[:max_tokens], skip_special_tokens=True) + "..."
    
    def _clean_entity_attributes(self, attrs: Dict[str, str]) -> str:
        # Fields to ignore (not useful for answering questions)
        ignore_keys = {"image", "image_size", "coordinates", "mapframe_wikidata", "website"}
        facts = []
        for key, value in attrs.items():
            if key in ignore_keys:
                continue  # skip irrelevant fields
            
            # Remove HTML tags (e.g., <br />) and Wikipedia markup from the value
            cleaned_val = value
            # Replace HTML line breaks with a comma + space (or just space)
            cleaned_val = cleaned_val.replace("<br />", ", ")
            cleaned_val = cleaned_val.replace("<br/>", ", ")
            # Remove Wiki link markup [[text|display]] or [[text]]
            # Replace '[[X|Y]]' with 'Y' and '[[X]]' with 'X'
            while '[[' in cleaned_val and ']]' in cleaned_val:
                start = cleaned_val.index('[[')
                end = cleaned_val.index(']]', start) + 2
                link_text = cleaned_val[start+2:end-2]  # text inside [[ ]]
                if '|' in link_text:
                    # Keep only the part after the pipe (display text)
                    link_text = link_text.split('|')[-1]
                cleaned_val = cleaned_val[:start] + link_text + cleaned_val[end:]

            while '{{' in cleaned_val and '}}' in cleaned_val:
                start = cleaned_val.index('{{')
                end = cleaned_val.index('}}', start) + 2
                # If it's a convert template, attempt to extract the primary value and unit
                segment = cleaned_val[start:end]
                if segment.lower().startswith("{{convert"):
                    parts = segment.strip("{}").split('|')
                    # Example: {{convert|870|ft|m|0|abbr=on}} -> parts = ["convert", "870", "ft", "m", "0", "abbr=on"]
                    if len(parts) >= 3:
                        number = parts[1]
                        unit = parts[2]
                        replacement = f"{number} {unit}"
                    else:
                        replacement = ""
                else:
                    replacement = ""
                cleaned_val = cleaned_val[:start] + replacement + cleaned_val[end:]
            cleaned_val = cleaned_val.strip().strip(',')
            if not cleaned_val:
                continue

            pretty_key = key.replace('_', ' ').capitalize()
            facts.append(f"{pretty_key}: {cleaned_val}")
        knowledge_text = "; ".join(facts)
        return knowledge_text
    
    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]]
    ) -> List[str]:

        batch_size = len(queries)
        responses: List[str] = [None] * batch_size  # to store final answers
        
# Step 1: Extract visual cues from all images using the VLM (vision-language model).
        image_summaries = self.batch_summarize_images(images)
        
# Step 2: Perform image search for each image to retrieve similar images and KG attributes.
        knowledge_list: List[str] = [None] * batch_size  # will store formatted knowledge for each query
        for i, (query, img) in enumerate(zip(queries, images)):
            results = self.search_pipeline(img, k=3)  # retrieve top-3 similar image results
            if not results or len(results) == 0:
                # No similar images / entity found in the knowledge base
                responses[i] = "I don't know."  # respond gracefully if we cannot find any info
                continue
            # Take the top result's first entity (most relevant match)
            top_result = results[0]
            entities = top_result.get('entities', [])
            if not entities or len(entities) == 0:
                responses[i] = "I don't know."
                continue
            entity = entities[0]
            attrs = entity.get('entity_attributes', {})
            if not attrs:
                responses[i] = "I don't know."
                continue
            # Clean and format the entity's attributes into a knowledge string
            knowledge_text = self._clean_entity_attributes(attrs)
            knowledge_text = self._truncate_to_tokens(knowledge_text, MAX_KNOWLEDGE_TOKENS)
            if not knowledge_text.strip():
                responses[i] = "I don't know."
            else:
                knowledge_list[i] = knowledge_text
        
        # Step 3: Compose answers using the vision-language model, grounded in the retrieved knowledge.
        # Prepare prompts for queries where we have knowledge available.
        gen_inputs = []
        idx_map = []  # map from generation input index back to original index
        for i, (query, knowledge) in enumerate(zip(queries, knowledge_list)):
            if responses[i] is not None:
                continue

            system_prompt = (
                "You are a helpful assistant that answers the user's question truthfully using the provided image and information. "
                "Here are some facts about the image:\n"
                f"{knowledge}\n"
                "Only use these facts to answer the question concisely and factually. If the facts are not sufficient, say \"I don't know.\""
            )
            # Construct the message list for the model
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [ {"type": "image"} ]},  # image placeholder
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
        
        # Generate answers for all prompts in gen_inputs as a batch
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
            # Extract text from outputs and assign to proper index in responses
            for output, orig_idx in zip(outputs, idx_map):
                answer = output.outputs[0].text.strip()
                # If model didn't produce any content or only whitespace, default to "I don't know."
                responses[orig_idx] = answer if answer else "I don't know."
        
        # For any query where knowledge was found but no response was generated, fill "I don't know."
        for i in range(batch_size):
            if responses[i] is None:
                responses[i] = "I don't know."
        
        return responses
