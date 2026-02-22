
import os
import argparse
import google.generativeai as genai

def generate_schematic(prompt, output_file="gemini_schematic.png"):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it via: $env:GOOGLE_API_KEY='your_key'")
        return

    genai.configure(api_key=api_key)
    
    # Priority list based on available models in environment
    # Prioritizing Nano Banana Pro and Gemini 2.5/Pro models for best SVG generation
    preferences = [
        'models/nano-banana-pro-preview',
        'models/gemini-2.5-flash',
        'models/gemini-2.5-pro',
        'models/gemini-2.0-flash',
        'models/gemini-1.5-pro',
        'models/gemini-pro'
    ]
    
    selected_model_name = None
    
    try:
        # Check availability
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        for pref in preferences:
            if pref in available_models:
                selected_model_name = pref
                break
        
        if not selected_model_name:
            # Fallback
            selected_model_name = available_models[0] if available_models else None
            
        if not selected_model_name:
            print("Error: No suitable generative models found.")
            return

        print(f"Generating schematic using: {selected_model_name}")
        model = genai.GenerativeModel(selected_model_name)
        
        # Ask Gemini to generate SVG code for the schematic
        svg_prompt = f"""
        Create a detailed, scientific schematic SVG diagram for the following description:
        "{prompt}"
        
        Requirements:
        - Use professional colors (scientific publication style).
        - Label all key components clearly.
        - The diagram should illustrate the vertical packing layers: Calcium Sulfate Slab at the bottom, a buffer zone, and a molecule packing vacuum region on top.
        - Output ONLY the raw SVG code. No markdown backticks.
        """
        
        response = model.generate_content(svg_prompt)
        svg_content = response.text
        
        # Clean up code blocks if present
        if "```svg" in svg_content:
            svg_content = svg_content.split("```svg")[1].split("```")[0].strip()
        elif "```xml" in svg_content:
            svg_content = svg_content.split("```xml")[1].split("```")[0].strip()
        elif "```" in svg_content:
            svg_content = svg_content.split("```")[1].split("```")[0].strip()
            
        # Save SVG with UTF-8 encoding
        svg_file = output_file.replace(".png", ".svg")
        with open(svg_file, "w", encoding="utf-8") as f:
            f.write(svg_content)
        
        print(f"✓ Success! SVG schematic saved to: {svg_file}")

        # Convert to PNG
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            
            print("Converting SVG to PNG...")
            drawing = svg2rlg(svg_file)
            
            # Use original output filename (ensure it ends in .png)
            if not output_file.lower().endswith('.png'):
                output_file += ".png"
                
            renderPM.drawToFile(drawing, output_file, fmt="PNG")
            print(f"✓ Success! PNG schematic saved to: {output_file}")
            
        except ImportError:
            print("Warning: svglib or reportlab not found. Skipping PNG conversion.")
        except Exception as e:
            print(f"Warning: Failed to convert SVG to PNG: {e}")
            
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Description of the schematic")
    parser.add_argument("-o", "--output", default="gemini_packing.png", help="Output filename")
    args = parser.parse_args()
    
    generate_schematic(args.prompt, args.output)
