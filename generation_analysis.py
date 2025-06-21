"""
Analysis tools for LLaDA generation history.
Provides functions to visualize the iterative diffusion process.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import textwrap
import os
import re

def analyze_token_finalization(history: Dict, tokenizer, verbose: bool = True) -> Dict:
    """
    Analyze when each token was finalized during generation.
    
    Args:
        history: Generation history from llada_generate
        tokenizer: The tokenizer used for generation
        verbose: Whether to print detailed analysis
    
    Returns:
        Dictionary with analysis results
    """
    mask_id = history['mask_id']
    prompt_length = history['prompt_length']
    states = history['states']
    step_info = history['step_info']
    finalized_at_step = history['finalized_at_step']
    
    # Get final state
    final_state = states[-1][0].cpu().numpy()
    
    # Decode tokens
    all_tokens = []
    for i, token_id in enumerate(final_state):
        if i < prompt_length:
            # Prompt token
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            all_tokens.append({
                'position': i,
                'token_id': token_id,
                'token_text': token_text,
                'type': 'prompt',
                'finalized_step': None
            })
        elif token_id != mask_id:
            # Generated token
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            finalized_step = finalized_at_step.get(i, 'unknown')
            all_tokens.append({
                'position': i,
                'token_id': token_id,
                'token_text': token_text,
                'type': 'generated',
                'finalized_step': finalized_step
            })
    
    # Create analysis
    analysis = {
        'total_tokens': len(all_tokens),
        'prompt_tokens': len([t for t in all_tokens if t['type'] == 'prompt']),
        'generated_tokens': len([t for t in all_tokens if t['type'] == 'generated']),
        'total_steps': len(step_info) - 1,  # Excluding final state
        'tokens': all_tokens,
        'finalization_timeline': {}
    }
    
    # Group tokens by finalization step
    for token in all_tokens:
        if token['type'] == 'generated' and token['finalized_step'] != 'unknown':
            step = token['finalized_step']
            if step not in analysis['finalization_timeline']:
                analysis['finalization_timeline'][step] = []
            analysis['finalization_timeline'][step].append(token)
    
    if verbose:
        print("=== TOKEN FINALIZATION ANALYSIS ===")
        print(f"Total tokens: {analysis['total_tokens']}")
        print(f"Prompt tokens: {analysis['prompt_tokens']}")
        print(f"Generated tokens: {analysis['generated_tokens']}")
        print(f"Total generation steps: {analysis['total_steps']}")
        print("\n=== FINALIZATION TIMELINE ===")
        
        for step in sorted(analysis['finalization_timeline'].keys()):
            tokens = analysis['finalization_timeline'][step]
            step_info_item = step_info[step] if step < len(step_info) else None
            block_info = f"(Block {step_info_item['block']}, Step {step_info_item['block_step']})" if step_info_item else ""
            
            print(f"\nStep {step} {block_info}:")
            for token in tokens:
                print(f"  Position {token['position']:2d}: '{token['token_text']}' (ID: {token['token_id']})")
        
        print("\n=== FULL SEQUENCE WITH FINALIZATION STEPS ===")
        full_text = ""
        for token in all_tokens:
            if token['type'] == 'prompt':
                full_text += f"[P]{token['token_text']}"
            else:
                step = token['finalized_step']
                full_text += f"[{step}]{token['token_text']}"
        print(f"Text: {full_text}")
        print("\nLegend: [P] = Prompt token, [N] = Generated at step N")
    
    return analysis

def create_generation_gif(history: Dict, tokenizer, output_path: str = "outputs/generation_evolution.gif", 
                         figsize: Tuple[int, int] = (14, 8), duration: int = 800,
                         wrap_width: int = 80) -> None:
    """
    Create an animated GIF showing the evolution of text generation.
    
    Args:
        history: Generation history from llada_generate
        tokenizer: The tokenizer used for generation
        output_path: Path to save the GIF
        figsize: Figure size (width, height)
        duration: Duration of each frame in milliseconds
        wrap_width: Characters per line for text wrapping
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    mask_id = history['mask_id']
    prompt_length = history['prompt_length']
    states = history['states']
    step_info = history['step_info']
    
    plt.style.use('default')
    colors = {
        'prompt': '#A23B72',      # Purple
        'finalized': '#2E86AB',   # Blue
        'masked': '#F18F01',      # Orange
        'background': '#FFFFFF',
        'text': '#333333'
    }
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor(colors['background'])
    
    def animate(frame):
        ax.clear()
        renderer = fig.canvas.get_renderer()
        
        current_state = states[frame][0].cpu().numpy()
        current_step_info = step_info[frame] if frame < len(step_info) else step_info[-1]
        mask_char = "█"

        # --- Text Processing ---
        # Decode the entire prompt at once to correctly handle special tokens.
        prompt_ids = current_state[:prompt_length]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

        # Clean up "user" and "assistant" markers left by some chat templates.
        prompt_text = re.sub(r'^\s*user\s*', '', prompt_text, flags=re.IGNORECASE).strip()
        prompt_text = re.sub(r'\s*assistant\s*$', '', prompt_text, flags=re.IGNORECASE).strip()

        # --- Rendering ---
        y_pos = 0.95
        line_height = 0.045
        
        # 1. Render User section
        ax.text(0.05, y_pos, "User:", transform=ax.transAxes, fontsize=13, fontfamily='monospace', fontweight='bold', va='top', color=colors['text'])
        y_pos -= line_height
        
        user_lines = textwrap.wrap(prompt_text, width=wrap_width, break_long_words=False, break_on_hyphens=False)
        for line in user_lines:
            display_line = line.replace('\\', '\\\\').replace('$', '\\$')
            ax.text(0.05, y_pos, display_line, transform=ax.transAxes, fontsize=12, fontfamily='monospace', va='top', color=colors['prompt'])
            y_pos -= line_height

        y_pos -= line_height * 0.5
        
        # 2. Render Assistant section
        ax.text(0.05, y_pos, "Assistant:", transform=ax.transAxes, fontsize=13, fontfamily='monospace', fontweight='bold', va='top', color=colors['text'])
        y_pos -= line_height

        # Create a list of (text, color) tuples for the assistant's response
        assistant_parts = []
        for token_id in current_state[prompt_length:]:
            if token_id == mask_id:
                assistant_parts.append({'text': mask_char, 'color': colors['masked']})
            else:
                # Use skip_special_tokens=True here to avoid printing things like '<s>'
                text = tokenizer.decode([token_id], skip_special_tokens=True)
                if text:
                    assistant_parts.append({'text': text, 'color': colors['finalized']})
        
        # Layout engine: render parts one by one, handling wrapping
        x_cursor = 0.05
        # Join parts to form words, preserving color info
        words_with_color = []
        for part in assistant_parts:
            # Treat each part as a "word" for layout purposes. This handles cases where tokens are punctuation.
            # Add a space to each word for layout, we will render word by word.
            words = (part['text'] + ' ').split(' ')
            for word in words:
                if word:
                    words_with_color.append({'text': word, 'color': part['color']})
        
        for item in words_with_color:
            word = item['text']
            color = item['color']
            
            # Render word and measure its width to position the next one
            # Add a space to the end for rendering.
            display_word = (word + ' ').replace('\\', '\\\\').replace('$', '\\$')
            
            text_obj = ax.text(x_cursor, y_pos, display_word, transform=ax.transAxes, fontsize=12, fontfamily='monospace', color=color, va='top')
            bbox = text_obj.get_window_extent(renderer=renderer)
            width_in_axes_coords = bbox.width / ax.get_window_extent().width
            
            if x_cursor + width_in_axes_coords > 0.95 and x_cursor > 0.05: # Wrap line
                x_cursor = 0.05
                y_pos -= line_height
                text_obj.set_position((x_cursor, y_pos))
                
                # Recalculate width after moving
                bbox = text_obj.get_window_extent(renderer=renderer)
                width_in_axes_coords = bbox.width / ax.get_window_extent().width

            x_cursor += width_in_axes_coords
            # Remove the temporary space from the word to avoid double spaces on wrap
            text_obj.set_text(display_word.rstrip(' '))

        # --- Info & Legend ---
        step_text = f"Step {current_step_info['global_step']}"
        if 'final' in current_step_info:
            step_text = "Generation Complete"
        ax.text(0.02, 0.02, step_text, transform=ax.transAxes, fontsize=9, color=colors['text'])
        
        legend_items = [("Masked", colors['masked']), ("Generated", colors['finalized']), ("Prompt", colors['prompt'])]
        x_cursor = 0.98
        for label, col in legend_items:
            label_text = f"● {label}"
            text_obj = ax.text(x_cursor, 0.98, label_text, transform=ax.transAxes, fontsize=9, color=col, ha='right', va='top')
            bbox = text_obj.get_window_extent(renderer=renderer)
            width_in_axes_coords = bbox.width / ax.get_window_extent().width
            x_cursor -= (width_in_axes_coords + 0.015)
        
        mask_count = np.sum(current_state == mask_id)
        final_count = len(current_state) - prompt_length - mask_count
        count_text = f"P: {prompt_length} | G: {final_count} | M: {mask_count}"
        ax.text(0.98, 0.02, count_text, transform=ax.transAxes, fontsize=9, color=colors['text'], ha='right')
        
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

    anim = FuncAnimation(fig, animate, frames=len(states), interval=duration)
    
    print(f"Creating GIF with {len(states)} frames...")
    fps = max(1, 1000 // duration)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    print(f"GIF saved to: {output_path}")
    
    plt.close(fig)

def create_prediction_evolution_gif(history: Dict, tokenizer, output_path: str = "outputs/prediction_evolution.gif",
                                  figsize: Tuple[int, int] = (14, 8), duration: int = 800,
                                  wrap_width: int = 80) -> None:
    """
    Create an animated GIF showing the evolution of text generation, with live predictions for masked tokens.
    
    Args:
        history: Generation history from llada_generate
        tokenizer: The tokenizer used for generation
        output_path: Path to save the GIF
        figsize: Figure size (width, height)
        duration: Duration of each frame in milliseconds
        wrap_width: Characters per line for text wrapping
    """
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    
    if 'predictions' not in history or not history['predictions']:
        print("Warning: Prediction history not found. Skipping prediction evolution GIF.")
        return

    mask_id = history['mask_id']
    prompt_length = history['prompt_length']
    states = history['states']
    predictions = history['predictions']
    step_info = history['step_info']
    
    # Set up color scheme
    plt.style.use('default')
    colors = {
        'prompt': '#A23B72',      # Purple
        'finalized': '#2E86AB',   # Blue
        'predicted': '#F18F01',    # Orange for predicted (not-yet-finalized) tokens
        'background': '#FFFFFF',   # White background
        'text': '#333333'         # Dark text
    }
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor(colors['background'])

    def get_token_segments(frame_idx: int) -> List[Dict]:
        """Group tokens into colored segments for rendering."""
        current_state = states[frame_idx][0].cpu().numpy()
        current_prediction = predictions[frame_idx][0].cpu().numpy()
        
        segments = []
        for i in range(len(current_state)):
            token_id = current_state[i]
            
            if i < prompt_length:
                color = colors['prompt']
                # Decode without skipping special tokens to clean up later
                text_to_render = tokenizer.decode([token_id])
                segments.append({'text': text_to_render, 'color': color})
            elif token_id != mask_id:
                color = colors['finalized']
                text_to_render = tokenizer.decode([token_id], skip_special_tokens=True)
                if text_to_render:
                    segments.append({'text': text_to_render, 'color': color})
            else: # Masked token
                color = colors['predicted']
                predicted_token_id = current_prediction[i]
                text_to_render = tokenizer.decode([predicted_token_id], skip_special_tokens=True)
                if text_to_render:
                    segments.append({'text': text_to_render, 'color': color})
        
        return segments

    def animate(frame):
        ax.clear()
        renderer = fig.canvas.get_renderer()
        
        current_state = states[frame][0].cpu().numpy()
        current_prediction = predictions[frame][0].cpu().numpy()
        current_step_info = step_info[frame] if frame < len(step_info) else step_info[-1]
        
        # --- Text Processing ---
        # 1. Decode prompt
        prompt_ids = current_state[:prompt_length]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

        # Clean up "user" and "assistant" markers left by some chat templates.
        prompt_text = re.sub(r'^\s*user\s*', '', prompt_text, flags=re.IGNORECASE).strip()
        prompt_text = re.sub(r'\s*assistant\s*$', '', prompt_text, flags=re.IGNORECASE).strip()

        # 2. Create segments for the assistant response
        assistant_segments = []
        for i in range(prompt_length, len(current_state)):
            token_id = current_state[i]
            
            if token_id != mask_id:
                color = colors['finalized']
                text = tokenizer.decode([token_id], skip_special_tokens=True)
                if text:
                    assistant_segments.append({'text': text, 'color': color})
            else: # Masked token
                color = colors['predicted']
                predicted_token_id = current_prediction[i]
                text = tokenizer.decode([predicted_token_id], skip_special_tokens=True)
                if text:
                    assistant_segments.append({'text': text, 'color': color})

        # --- Rendering ---
        y_pos = 0.95
        line_height = 0.045

        # 1. Render User section
        ax.text(0.05, y_pos, "User:", transform=ax.transAxes, fontsize=13, fontfamily='monospace', fontweight='bold', va='top', color=colors['text'])
        y_pos -= line_height
        
        user_lines = textwrap.wrap(prompt_text, width=wrap_width, break_long_words=False, break_on_hyphens=False)
        for line in user_lines:
            display_line = line.replace('\\', '\\\\').replace('$', '\\$')
            ax.text(0.05, y_pos, display_line, transform=ax.transAxes, fontsize=12, fontfamily='monospace', va='top', color=colors['prompt'])
            y_pos -= line_height

        y_pos -= line_height * 0.5
        
        # 2. Render Assistant section
        ax.text(0.05, y_pos, "Assistant:", transform=ax.transAxes, fontsize=13, fontfamily='monospace', fontweight='bold', va='top', color=colors['text'])
        y_pos -= line_height

        x_cursor = 0.05
        words_with_color = []
        for item in assistant_segments:
            words = (item['text'] + ' ').split(' ')
            for word in words:
                if word:
                    words_with_color.append({'text': word, 'color': item['color']})

        for item in words_with_color:
            word = item['text']
            color = item['color']
            display_word = (word + ' ').replace('\\', '\\\\').replace('$', '\\$')
            
            text_obj = ax.text(x_cursor, y_pos, display_word, transform=ax.transAxes, fontsize=12, fontfamily='monospace', color=color, va='top')
            bbox = text_obj.get_window_extent(renderer=renderer)
            width_in_axes_coords = bbox.width / ax.get_window_extent().width
            
            if x_cursor + width_in_axes_coords > 0.95 and x_cursor > 0.05:
                x_cursor = 0.05
                y_pos -= line_height
                text_obj.set_position((x_cursor, y_pos))
                
                bbox = text_obj.get_window_extent(renderer=renderer)
                width_in_axes_coords = bbox.width / ax.get_window_extent().width

            x_cursor += width_in_axes_coords
            text_obj.set_text(display_word.rstrip(' '))

        # --- Info & Legend ---
        step_text = f"Step {current_step_info['global_step']}"
        if 'final' in current_step_info:
            step_text = "Generation Complete"
        ax.text(0.02, 0.02, step_text, transform=ax.transAxes, fontsize=9, color=colors['text'])

        legend_items = [("Predicted", colors['predicted']), ("Generated", colors['finalized']), ("Prompt", colors['prompt'])]
        x_cursor = 0.98
        for label, col in legend_items:
            label_text = f"● {label}"
            text_obj = ax.text(x_cursor, 0.98, label_text, transform=ax.transAxes, fontsize=9, color=col, ha='right', va='top')
            bbox = text_obj.get_window_extent(renderer=renderer)
            width_in_axes_coords = bbox.width / ax.get_window_extent().width
            x_cursor -= (width_in_axes_coords + 0.015)
        
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

    print(f"Creating Prediction GIF with {len(states)} frames...")
    anim = FuncAnimation(fig, animate, frames=len(states), interval=duration)
    fps = max(1, 1000 // duration)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Prediction GIF saved to: {output_path}")

def create_finalization_heatmap_gif(history: Dict, tokenizer, output_path: str = "outputs/finalization_heatmap.gif",
                                    figsize: Tuple[int, int] = (14, 8), duration: int = 800,
                                    wrap_width: int = 80) -> None:
    """
    Creates a GIF where finalized tokens are colored by their finalization step using a blue color scale.
    
    Args:
        history: Generation history from llada_generate
        tokenizer: The tokenizer used for generation
        output_path: Path to save the GIF
        figsize: Figure size (width, height)
        duration: Duration of each frame in milliseconds
        wrap_width: Characters per line for text wrapping
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if 'predictions' not in history or not history['predictions']:
        print("Warning: Prediction history not found. Skipping heatmap GIF.")
        return

    mask_id = history['mask_id']
    prompt_length = history['prompt_length']
    states = history['states']
    predictions = history['predictions']
    step_info = history['step_info']
    finalized_at_step = history['finalized_at_step']
    total_steps = len(states) - 1

    # Use a sequential blue colormap for the heatmap effect
    # Lighter colors for earlier steps (e.g., Blues_r would be dark to light)
    colormap = plt.cm.get_cmap('Blues') 

    plt.style.use('default')
    colors = {
        'prompt': '#A23B72',      # A distinct color for the prompt (Purple)
        'predicted': '#F18F01',   # Orange for predicted
        'background': '#FFFFFF',
        'text': '#333333'
    }

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor(colors['background'])

    def animate(frame):
        ax.clear()
        renderer = fig.canvas.get_renderer()
        
        current_state = states[frame][0].cpu().numpy()
        current_prediction = predictions[frame][0].cpu().numpy()
        current_step_info = step_info[frame] if frame < len(step_info) else step_info[-1]
        
        # --- Text Processing ---
        prompt_ids = current_state[:prompt_length]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        prompt_text = re.sub(r'^\s*user\s*', '', prompt_text, flags=re.IGNORECASE).strip()
        prompt_text = re.sub(r'\s*assistant\s*$', '', prompt_text, flags=re.IGNORECASE).strip()

        assistant_segments = []
        for i in range(prompt_length, len(current_state)):
            token_id = current_state[i]
            is_finalized_now = token_id != mask_id
            
            if is_finalized_now:
                finalization_step = finalized_at_step.get(i, 0)
                # Normalize step to a darker part of the colormap for better readability.
                # The range is mapped from [0.0, 1.0] to [0.4, 1.0]
                norm_step = (finalization_step / max(1, total_steps)) * 0.6 + 0.4
                color = colormap(norm_step)
                text = tokenizer.decode([token_id], skip_special_tokens=True)
            else: # Masked token, show prediction
                color = colors['predicted']
                predicted_token_id = current_prediction[i]
                text = tokenizer.decode([predicted_token_id], skip_special_tokens=True)

            if text:
                assistant_segments.append({'text': text, 'color': color})

        # --- Rendering ---
        y_pos = 0.95
        line_height = 0.045

        # User section
        ax.text(0.05, y_pos, "User:", transform=ax.transAxes, fontsize=13, fontfamily='monospace', fontweight='bold', va='top', color=colors['text'])
        y_pos -= line_height
        user_lines = textwrap.wrap(prompt_text, width=wrap_width, break_long_words=False, break_on_hyphens=False)
        for line in user_lines:
            ax.text(0.05, y_pos, line, transform=ax.transAxes, fontsize=12, fontfamily='monospace', va='top', color=colors['prompt'])
            y_pos -= line_height
        y_pos -= line_height * 0.5
        
        # Assistant section
        ax.text(0.05, y_pos, "Assistant:", transform=ax.transAxes, fontsize=13, fontfamily='monospace', fontweight='bold', va='top', color=colors['text'])
        y_pos -= line_height

        x_cursor = 0.05
        words_with_color = []
        for item in assistant_segments:
            words = (item['text'] + ' ').split(' ')
            for word in words:
                if word:
                    words_with_color.append({'text': word, 'color': item['color']})
        
        for item in words_with_color:
            word = item['text']
            color = item['color']
            display_word = (word + ' ').replace('\\', '\\\\').replace('$', '\\$')
            
            text_obj = ax.text(x_cursor, y_pos, display_word, transform=ax.transAxes, fontsize=12, fontfamily='monospace', color=color, va='top')
            bbox = text_obj.get_window_extent(renderer=renderer)
            width_in_axes_coords = bbox.width / ax.get_window_extent().width
            
            if x_cursor + width_in_axes_coords > 0.95 and x_cursor > 0.05:
                x_cursor = 0.05
                y_pos -= line_height
                text_obj.set_position((x_cursor, y_pos))
                bbox = text_obj.get_window_extent(renderer=renderer)
                width_in_axes_coords = bbox.width / ax.get_window_extent().width

            x_cursor += width_in_axes_coords
            text_obj.set_text(display_word.rstrip(' '))

        # --- Info & Legend ---
        step_text = f"Step {current_step_info['global_step']}"
        if 'final' in current_step_info: step_text = "Generation Complete"
        ax.text(0.02, 0.02, step_text, transform=ax.transAxes, fontsize=9, color=colors['text'])

        # New legend at the top right
        ax.text(0.98, 0.98, "● Finalized (Light → Dark Blue)", transform=ax.transAxes, fontsize=9, color=colormap(0.9), ha='right')
        ax.text(0.98, 0.95, f"● Predicted", transform=ax.transAxes, fontsize=9, color=colors['predicted'], ha='right')
        ax.text(0.98, 0.92, f"● Prompt", transform=ax.transAxes, fontsize=9, color=colors['prompt'], ha='right')
        
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

    print(f"Creating Finalization Heatmap GIF with {len(states)} frames...")
    anim = FuncAnimation(fig, animate, frames=len(states), interval=duration)
    fps = max(1, 1000 // duration)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Finalization Heatmap GIF saved to: {output_path}")

def create_static_visualization(analysis: Dict, tokenizer, output_path: str = "outputs/generation_analysis.png",
                               figsize: Tuple[int, int] = (16, 10)) -> None:
    """
    Create a static heatmap visualization of token finalization steps.
    
    Args:
        analysis: Analysis dictionary from analyze_token_finalization.
        tokenizer: The tokenizer used for generation.
        output_path: Path to save the PNG file.
        figsize: Figure size for the plot.
    """
    
    generated_tokens = [t for t in analysis['tokens'] if t['type'] == 'generated']
    if not generated_tokens:
        print("No generated tokens to visualize.")
        return

    # Prepare data for heatmap
    num_gen_tokens = len(generated_tokens)
    max_steps = analysis['total_steps']
    
    heatmap_data = np.full((max_steps + 1, num_gen_tokens), np.nan)
    
    for i, token in enumerate(generated_tokens):
        step = token['finalized_step']
        if step != 'unknown' and step is not None:
            heatmap_data[step, i] = step

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(heatmap_data, ax=ax, cmap='viridis', cbar_kws={'label': 'Finalization Step'})
    ax.set_xlabel('Token Position (Generated Only)')
    ax.set_ylabel('Generation Step')
    ax.set_title('Token Finalization Timeline')
    
    # Add token text annotations
    for i, token in enumerate(generated_tokens):
        step = token['finalized_step']
        if step != 'unknown' and step >= 0:
            text = token['token_text'][:8] + "..." if len(token['token_text']) > 8 else token['token_text']
            # Escape backslashes and dollar signs to prevent matplotlib from interpreting as LaTeX/MathText
            display_text = text.replace('\\', '\\\\').replace('$', '\\$')
            ax.text(i + 0.5, step + 0.5, display_text, ha='center', va='center', 
                    fontsize=8, color='white', weight='bold')
    
    fig.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Static visualization saved to: {output_path}")

def print_generation_summary(history: Dict, tokenizer) -> None:
    """
    Print a concise summary of the generation process.
    
    Args:
        history: Generation history from llada_generate
        tokenizer: The tokenizer used for generation
    """
    analysis = analyze_token_finalization(history, tokenizer, verbose=False)
    
    print("=== GENERATION SUMMARY ===")
    print(f"Total steps: {analysis['total_steps']}")
    print(f"Generated tokens: {analysis['generated_tokens']}")
    print(f"Average tokens per step: {analysis['generated_tokens'] / analysis['total_steps']:.2f}")
    
    # Show progression
    print("\n=== STEP-BY-STEP PROGRESSION ===")
    timeline = analysis['finalization_timeline']
    for step in sorted(timeline.keys()):
        tokens = timeline[step]
        token_texts = [t['token_text'] for t in tokens]
        print(f"Step {step:2d}: {len(tokens)} tokens - {', '.join(repr(t) for t in token_texts)}")

# Convenience function to run all analyses
def full_analysis(history: Dict, tokenizer, base_name: str = "outputs/generation") -> None:
    """
    Run a full analysis suite on the generation history.
    """
    print("\n=== Running Full LLaDA Generation Analysis ===")
    
    # 1. Analyze token finalization
    print("\n1. Analyzing token finalization...")
    analysis = analyze_token_finalization(history, tokenizer, verbose=False) # Keep console clean
    print_generation_summary(history, tokenizer)
    
    # 2. Create static visualization
    print("\n2. Creating static visualization...")
    static_path = f"{base_name}_analysis.png"
    create_static_visualization(analysis, tokenizer, output_path=static_path)

    # 3. Create standard animated GIF
    print("\n3. Creating standard animated GIF...")
    gif_path = f"{base_name}_evolution.gif"
    create_generation_gif(history, tokenizer, output_path=gif_path)

    # 4. Create prediction evolution GIF
    print("\n4. Creating prediction evolution GIF...")
    pred_gif_path = f"{base_name}_prediction_evolution.gif"
    create_prediction_evolution_gif(history, tokenizer, output_path=pred_gif_path)

    # 5. Create finalization heatmap GIF
    print("\n5. Creating finalization heatmap GIF...")
    finalization_heatmap_path = f"{base_name}_finalization_heatmap.gif"
    create_finalization_heatmap_gif(history, tokenizer, output_path=finalization_heatmap_path)

    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print(f"- {static_path}: Static visualization")
    print(f"- {gif_path}: Animated generation process (masked tokens)")
    print(f"- {pred_gif_path}: Animated generation process (predicted tokens)")
    print(f"- {finalization_heatmap_path}: Finalization heatmap GIF") 