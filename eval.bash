# Quick test (2 samples)
# inspect eval llada_inspect/gpqa.py --model llada/llada-1.5 --limit 2

# inspect eval llada_inspect/gsm8k.py --model llada/llada-1.5 --limit 100 -T fewshot=false

# With chain-of-thought disabled
inspect eval llada_inspect/gsm8k.py --limit 100 --model llada/llada-1.5 -T cot=false -T fewshot=false -M gen_length=32 -M remasking=random

# MMADA
# inspect eval llada_inspect/gsm8k.py --model mmada/mmada-8b-mixcot --limit 100 -T fewshot=false