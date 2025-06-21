import os
from llada_inference import generate_text
from generation_analysis import create_generation_gif, create_prediction_evolution_gif, create_finalization_heatmap_gif


prompt = """Please fill in the blanks. Premise: "A soccer team is practicing on a field." Noisy Hypothesis: "Some [BLANK] are playing a sport."""

print("Generating text with history for GIF creation...")

generated_text, history, tokenizer = generate_text(
    prompt, 
    gen_length=64, 
    steps=32, 
    remasking='random',
    block_length=64, 
    save_history=True
)

print("GENERATED TEXT:")
print(generated_text)
print()

print("Creating standard generation GIF (with masked tokens)...")
create_generation_gif(
    history, 
    tokenizer, 
    output_path="outputs/generation_evolution.gif",
    duration=200 
)

print("\nCreating prediction evolution GIF (with predicted tokens)...")
create_prediction_evolution_gif(
    history,
    tokenizer,
    output_path="outputs/prediction_evolution.gif",
    duration=200
)

print("\nCreating finalization heatmap GIF...")
create_finalization_heatmap_gif(
    history,
    tokenizer,
    output_path="outputs/finalization_heatmap.gif",
    duration=200
)
print("\nGIFs have been saved in the 'outputs' directory.") 