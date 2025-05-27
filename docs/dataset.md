# CRAG-MM Dataset Documentation 🚀

**CRAG-MM (Comprehensive RAG Benchmark for Multi-modal, Multi-turn)** is a factual visual question-answering dataset that focuses on real-world images and conversation-based question answering. It is designed to help you evaluate and train **retrieval-augmented generation (RAG)** systems for both **single-turn** and **multi-turn** scenarios.

CRAG-MM is available on HuggingFace at the following links:
- **Single-turn**: [https://huggingface.co/datasets/crag-mm-2025/crag-mm-single-turn-public](https://huggingface.co/datasets/crag-mm-2025/crag-mm-single-turn-public)
- **Multi-turn**: [https://huggingface.co/datasets/crag-mm-2025/crag-mm-multi-turn-public](https://huggingface.co/datasets/crag-mm-2025/crag-mm-multi-turn-public)

---

## 1. Dataset Highlights

1. **Images**: A mixture of egocentric images captured from RayBan Meta Smart Glasses and images from publicly available sources.  
2. **Domains**: 13 total, ranging from `shopping` to `food` to `math and science`, aiming to test broad knowledge coverage.  
3. **Query Categories**: Simple recognition, multi-hop reasoning, comparison, aggregation, and more—covering diverse cognitive tasks.  
4. **Turns**: The dataset includes examples with just one Q&A pair (**single-turn**) as well as more extended dialogues with multiple Q&A pairs about the same image (**multi-turn**).  
5. **Image Quality Variations**: e.g. `normal`, `low light`, `blurred`, ensuring robustness to real-world conditions.

---

## 2. Data Splits

In publicly released `v0.1.1` of the CRAG-MM dataset, you can find the following splits:
- **validation**

The **sample** split is now deprecated. 

---

## 3. Data Structure

### Single-Turn Format

An example entry (one Q&A about a single image) may look like:

```json
{
  "session_id": "string",
  "image": Image(),
  "image_url": "string",
  "image_quality": "string",
  "turns": [
    {
      "interaction_id": "string",
      "domain": "string",
      "query_category": "string",
      "dynamism": "string",
      "query": "string",
    }
  ],
  "answers": [
    {
      "interaction_id": "string",
      "ans_full": "string"
    }
  ]
}
```
- **`session_id`**: Unique identifier for this example.  
- **`image`**: The loaded PIL Image for the single turn. Can be None
- **`ìmage_url`**: The image_url for conversations where `image` is set to None. 
- **`turns`**: Contains exactly one turn in single-turn format.  
  - **`interaction_id`** links to the matching answer in `answers`.  
  - **`domain`**, **`query_category`**, **`dynamism`**, **`image_quality`**: Categorical labels describing the question and environment.  
- **`answers`**: The list (length 1) containing the ground-truth answer.  

### Multi-Turn Format

An example entry (conversation with multiple Q&As on the same image) may look like:

```json
{
  "session_id": "string",
  "image": Image(),
  "image_url": "string",
  "image_quality": "string",
  "turns": [
    {
      "interaction_id": "string",
      "domain": "string", 
      "query_category": "string",
      "dynamism": "string",
      "query": "string",
    },
    ...
  ],
  "answers": [
    {
      "interaction_id": "string",
      "ans_full": "string"
    },
    ...
  ]
}
```
- **`session_id`**: Unique identifier for the entire conversation.  
- **`image`**: One image shared across all turns.  
- **`ìmage_url`**: The image_url for conversations where `image` is set to None. 
- **`turns`**: Each user query is listed as a separate object in this array.  
- **`interaction_id`** links to the matching answer in `answers`.  
- **`domain`**, **`query_category`**, **`dynamism`**, **`image_quality`**: Categorical labels describing the question and environment.  
- **`answers`**: Ground-truth answers, matching by `interaction_id`.

---

## 4. Accessing the Dataset

```python
from datasets import load_dataset
# For single-turn dataset
dataset = load_dataset("crag-mm-2025/crag-mm-single_turn-public", revision="v0.1.1")
# For multi-turn dataset
dataset = load_dataset("crag-mm-2025/crag-mm-multi_turn-public", revision="v0.1.1")
# View available splits
print(f"Available splits: {', '.join(dataset.keys())}")
# Access examples
example = dataset["validation"][0]
print(f"Session ID: {example['session_id']}")
print(f"Image: {example['image']}")
print(f"Image URL: {example['image_url']}")
"""
Note: Either 'image' or 'image_url' will be provided in the dataset, but not necessarily both.
When the actual image cannot be included, only the image_url will be available.
The evaluation servers will nevertheless always include the loaded 'image' field.
"""
# Show image
import matplotlib.pyplot as plt
plt.imshow(example['image'])
```

More details about the usage are available on the Huggingface dataset card. 

---

## 5. License & Citation

- **License**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0).  
- **Citation**: If you use CRAG-MM in your research, please cite:

```bibtex
@inproceedings{crag-mm-2025,
  title = {CRAG-MM: A Comprehensive RAG Benchmark for Multi-modal, Multi-turn Question Answering},
  author = {CRAG-MM Team},
  year = {2025},
  url = {https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025}
}
```

