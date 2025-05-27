export CRAG_CACHE_DIR=/srv/scratch/CRUISE/z5544297/.cache


python local_evaluation.py     --dataset-type single-turn     --split validation     --num-conversations 20     --display-conversations 3 --suppress-web-search-api   --eval-model gpt-4o-mini --output-dir result/

# python local_evaluation.py     --dataset-type single-turn     --split validation     --num-conversations 50     --display-conversations 3    --suppress-web-search-api   --eval-model None --output-dir result/

# python local_evaluation.py --dataset-type single-turn --split public_test --num-conversations -1 --suppress-web-search-api

# python local_evaluation.py --dataset-type multi-turn --split validation --num-conversations -1 --suppress-web-search-api