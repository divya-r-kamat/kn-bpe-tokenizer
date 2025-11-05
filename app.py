"""
Gradio app for Kannada BPE Tokenizer visualization.
Usage: python app.py
"""
import gradio as gr
import re
from tokenizer import KannadaBPETokenizer


# Color palette for tokens
COLORS = [
    '#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF',
    '#FFB3E6', '#E6B3FF', '#FFE6B3', '#B3FFE6', '#B3E6FF',
    '#FFD1DC', '#FFE4B5', '#E6FFB3', '#B3FFF0', '#D1B3FF',
    '#FFCCE6', '#FFCCB3', '#FFFFCC', '#CCFFE6', '#CCE6FF'
]


def tokenize_and_visualize(text: str, tokenizer: KannadaBPETokenizer):
    """Tokenize text and create HTML visualization with hover effects."""
    if not text.strip():
        return (
            "<p style='color: gray; font-size: 16px;'>Enter some Kannada text to tokenize...</p>",
            "<p style='color: gray;'>Token count will appear here</p>",
            "<p style='color: gray;'>Token IDs will appear here</p>"
        )
    
    # Get token IDs
    token_ids = tokenizer.encode(text)
    
    # Create a mapping of unique token IDs to colors
    unique_token_ids = []
    token_id_to_color = {}
    color_idx = 0
    
    for tid in token_ids:
        if tid not in token_id_to_color:
            token_id_to_color[tid] = COLORS[color_idx % len(COLORS)]
            unique_token_ids.append(tid)
            color_idx += 1
    
    # Build token visualization
    html_parts = ['<div style="font-size: 20px; line-height: 2.8; font-family: Noto Sans Kannada, Arial, sans-serif; padding: 15px; background: #f9f9f9; border-radius: 8px;">']
    
    # Track position in original text
    chunks = re.findall(tokenizer.pattern, text)
    token_idx = 0
    token_info_list = []
    
    for chunk in chunks:
        chunk_tokens = tokenizer._apply_bpe(chunk)
        
        for tid in chunk_tokens:
            token_text = tokenizer.vocab[tid]
            color = token_id_to_color[tid]  # Use consistent color for same token ID
            
            # Store token info for display
            token_info_list.append((token_idx, tid, token_text, color))
            
            # Create span with hover effect
            html_parts.append(
                f'<span class="token-span" data-token-idx="{token_idx}" '
                f'style="background-color: {color}; padding: 4px 8px; '
                f'margin: 2px; border-radius: 4px; cursor: pointer; '
                f'transition: all 0.2s; display: inline-block; font-weight: 500;" '
                f'title="Token #{token_idx}&#10;Token ID: {tid}&#10;Text: {repr(token_text)}">'
                f'{token_text}</span>'
            )
            
            token_idx += 1
    
    html_parts.append('</div>')
    
    # Add custom CSS for better hover effects
    css = """
    <style>
        .token-span:hover {
            transform: scale(1.15);
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            z-index: 100;
        }
    </style>
    """
    
    tokens_html = css + ''.join(html_parts)
    
    # Create token count display
    count_html = f"""
    <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 12px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h2 style="color: white; margin: 0; font-size: 48px; font-weight: bold;">{len(token_ids)}</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 18px;">Total Tokens</p>
        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0; font-size: 14px;">{len(unique_token_ids)} Unique Tokens</p>
    </div>
    """
    
    # Create token IDs display
    token_ids_html = ['<div style="padding: 15px; background: #f0f0f0; border-radius: 8px; max-height: 400px; overflow-y: auto;">']
    token_ids_html.append('<table style="width: 100%; border-collapse: collapse; font-family: monospace;">')
    token_ids_html.append('<tr style="background: #667eea; color: white;"><th style="padding: 10px; text-align: left;">Index</th><th style="padding: 10px; text-align: left;">Token ID</th><th style="padding: 10px; text-align: left;">Token</th><th style="padding: 10px; text-align: left;">Color</th></tr>')
    
    for idx, tid, ttext, color in token_info_list:
        token_ids_html.append(
            f'<tr style="border-bottom: 1px solid #ddd;">'
            f'<td style="padding: 8px;">{idx}</td>'
            f'<td style="padding: 8px; font-weight: bold;">{tid}</td>'
            f'<td style="padding: 8px;">{repr(ttext)}</td>'
            f'<td style="padding: 8px;"><span style="background-color: {color}; padding: 4px 12px; border-radius: 4px; display: inline-block;">&nbsp;</span></td>'
            f'</tr>'
        )
    
    token_ids_html.append('</table></div>')
    
    return tokens_html, count_html, ''.join(token_ids_html)


def create_app(tokenizer: KannadaBPETokenizer):
    """Create Gradio interface."""
    
    def process_text(text):
        tokens_html, count_html, ids_html = tokenize_and_visualize(text, tokenizer)
        return tokens_html, count_html, ids_html
    
    with gr.Blocks(title="Kannada BPE Tokenizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🔤 Kannada BPE Tokenizer
            Visualize how Kannada text is tokenized using Byte Pair Encoding (BPE)
            """
        )
        
        with gr.Row():
            # Right Column - Input Section
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="Input Text",
                    placeholder="ನಮಸ್ಕಾರ, ಇದು ಕನ್ನಡ ಟೋಕನೈಜರ್ ಆಗಿದೆ",
                    lines=6,
                    max_lines=15
                )
                
                tokenize_btn = gr.Button("🔍 Tokenize", variant="primary", size="lg")
                
                gr.Markdown("### 📝 Examples")
                gr.Examples(
                    examples=[
                        ["ನಮಸ್ಕಾರ, ಇದು ಕನ್ನಡ ಟೋಕನೈಜರ್ ಆಗಿದೆ"],
                        ["ಅವನು ಬರುತ್ತಿದ್ದಾನೆ ಎಂದು ನನಗೆ ಗೊತ್ತು."],
                        ["ಅವಳು ಕೆಲಸ ಮುಗಿಸಿದ ನಂತರ ಮನೆಗೆ ಹೋದಳು."],
                        ["ನಾನು ಸೂರ್ಯ ಮುಳುಗುವುದಾದ ಯಾವಾಗ ಹೊರಗಡೆ ಹೋದೆನು."],
                        ["ಅವಳು ಮನೆಗೆ ಬಾರದ ಕಾರಣ, ಏಕೆಂದರೆ ಅವಳು ಕೆಲಸದಲ್ಲಿ ಬ್ಯುಸಿಯಾಗಿದ್ದಳು."],
                    ],
                    inputs=[input_text]
                )
            # Left Column - Output Section
            with gr.Column(scale=1):
                token_count = gr.HTML(
                    label="Token Count",
                    value="<p style='color: gray;'>Token count will appear here</p>"
                )
                
                output_html = gr.HTML(
                    label="Tokens (Hover to see details)"
                )
                
                token_ids_display = gr.HTML(
                    label="Token IDs",
                    value="<p style='color: gray;'>Token IDs will appear here</p>"
                )
        
        gr.Markdown(
            """
            ### ℹ️ How to use:
            - Enter or select Kannada text in the input area
            - Click "Tokenize" or press Enter
            - **Left panel** shows:
              - Token count at the top
              - Colored tokens (hover to see details)
              - Complete token ID table at the bottom
            - **Identical tokens share the same color** - easily spot repeated tokens!
            """
        )
        
        # Set up event handlers
        tokenize_btn.click(
            fn=process_text,
            inputs=[input_text],
            outputs=[output_html, token_count, token_ids_display]
        )
        
        input_text.submit(
            fn=process_text,
            inputs=[input_text],
            outputs=[output_html, token_count, token_ids_display]
        )
    
    return demo


def main():
    # Initialize tokenizer
    tokenizer = KannadaBPETokenizer()
    
    # Load pre-trained vocabulary
    try:
        tokenizer.load_vocab("model/vocab.json")
        print("✓ Loaded vocabulary successfully")
    except FileNotFoundError:
        print("ERROR: Vocabulary file not found at 'model/vocab.json'")
        print("Please run 'python train.py' first to train and save the tokenizer.")
        return
    
    # Create and launch app
    demo = create_app(tokenizer)
    demo.launch(share=True)


if __name__ == "__main__":
    main()
