import gradio as gr
import google.generativeai as genai
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# API Key
gemini_api_key = None
clothing_items = []  # (image, category)

# Load CLIP Model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Set Gemini API Key
def set_gemini_api_key(key):
    global gemini_api_key
    gemini_api_key = key
    genai.configure(api_key=key)
    try:
        models = genai.list_models()
        available_models = [model.name for model in models]
        return f"Gemini API key set! Available models: {available_models}"
    except Exception as e:
        return f"An error occurred: {e}"

# Upload and Recognize Clothing Type
def upload_clothing(image):  # <- receives PIL.Image now
    if image is not None:
        image = image.convert('RGBA')  # ensure transparency
        inputs = clip_processor(text=["shirt", "pants", "dress", "skirt", "coat", "shoes"],
                                images=image, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).tolist()[0]
        categories = ["Shirt", "Pants", "Dress", "Skirt", "Coat", "Shoes"]
        recognized_item = categories[probs.index(max(probs))]
        clothing_items.append((image, recognized_item))
        return image, f"{recognized_item} uploaded! Total items: {len(clothing_items)}"
    return None, "Please upload a valid image."

# Chat and Suggest Based on Weather and User Input
def chat_weather_outfit(user_input, weather_info):
    if not gemini_api_key:
        return "Please enter your API key!", None, [], []

    prompt = "You are a fashion assistant."
    prompt += "\nHere are the available clothing items:\n"
    for i, (_, item) in enumerate(clothing_items):
        prompt += f"Item {i+1}: {item}\n"
    prompt += f"\nCurrent weather: {weather_info}"
    prompt += f"\nUser input: {user_input}"
    prompt += "\nSuggest the best single outfit combination only, and clearly state the clothing categories (e.g., Shirt, Pants, Shoes)."

    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        suggestion = response.text.strip().lower()

        selected_images = []
        selected_labels = []
        used_categories = set()

        for img, label in clothing_items:
            if label.lower() in suggestion and label not in used_categories:
                selected_images.append(img)
                selected_labels.append(label)
                used_categories.add(label)

        return response.text.strip(), selected_images, selected_labels
    except Exception as e:
        return f"An error occurred: {e}", [], []

# Basic chatbot for general fashion Q&A
def fashion_chatbot(user_message):
    if not gemini_api_key:
        return "Please enter your API key!"
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(user_message)
        return response.text.strip()
    except Exception as e:
        return f"An error occurred: {e}"

# Gradio Interface
with gr.Blocks(theme='hmb/amethyst') as app:
    gr.Markdown("# 👗 AI Fashion Bot - Weather-Based Outfit Suggestions ")

    with gr.Row():
        gemini_api_key_input = gr.Textbox(label="Gemini API Key", type="password")
        gemini_api_key_button = gr.Button("Set API Key")
        api_status = gr.Textbox(label="Status")
        gemini_api_key_button.click(fn=set_gemini_api_key, inputs=gemini_api_key_input, outputs=api_status)

    with gr.Row():
        upload = gr.Image(type="pil", label="Upload Clothing Item")  # ✅ changed here
        upload_button = gr.Button("Upload")
        image_preview = gr.Image(label="Clothing Preview")
        upload_status = gr.Textbox(label="Upload Status")
        upload_button.click(fn=upload_clothing, inputs=upload, outputs=[image_preview, upload_status])

    gr.Markdown("---")

    with gr.Row():
        weather_input = gr.Textbox(label="Current Weather", placeholder="e.g., Sunny and 25°C")
        chat_input = gr.Textbox(label="Your Style Request", placeholder="Suggest something comfy and stylish")
        chat_button = gr.Button("Get Outfit Suggestion")

    suggestion_output = gr.Textbox(label="AI Suggestion", interactive=False, lines=6)
    selected_images_gallery = gr.Gallery(label="Clothing Items for Suggested Outfit", columns=3, height="auto")
    selected_labels_output = gr.Textbox(label="Item Categories in Suggestion", interactive=False)

    chat_button.click(fn=chat_weather_outfit,
                      inputs=[chat_input, weather_input],
                      outputs=[suggestion_output, selected_images_gallery, selected_labels_output])

    gr.Markdown("---")

    gr.Markdown("### 🧠 Chat with your AI Fashion Friend")
    user_question = gr.Textbox(label="Ask something fashion-related", placeholder="Can I wear boots in summer?")
    chatbot_response = gr.Textbox(label="Bot's Answer", interactive=False, lines=4)
    ask_button = gr.Button("Ask")
    ask_button.click(fn=fashion_chatbot, inputs=user_question, outputs=chatbot_response)

# Launch App
app.launch(share=True)
