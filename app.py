import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    system_message = "Hello! I'm the Tech Troubleshooter 🖥️. Please describe the technical issue you're facing, and I'll do my best to help you resolve it. Whether it's troubleshooting software problems, hardware issues, or general tech queries, I'm here to assist you!"
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="Hello! I'm the Tech Troubleshooter 🖥️. Please describe the technical issue you're facing, and I'll do my best to help you resolve it. Whether it's troubleshooting software problems, hardware issues, or general tech queries, I'm here to assist you!", label="System Message", lines=3),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Maximum Tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (Nucleus Sampling)",
        ),
    ],

    examples=[
        ["My computer is running slow. How can I speed it up? 🖥️"],
        ["I'm getting an error message when I try to install software. What should I do? 💻"],
        ["How do I connect my printer wirelessly to my computer? 🖨️"],
    ],
    title='Tech Troubleshooter 🖥️',
    description='''<div style="text-align: right; font-family: Arial, sans-serif; color: #333;">
                   <h2>Welcome to the Tech Troubleshooter 🖥️</h2>
                   <p style="font-size: 16px; text-align: right;">Please describe the technical issue you're facing, and I'll do my best to provide assistance.</p>
                   <p style="text-align: right;"><strong>Examples:</strong></p>
                   <ul style="list-style-type: disc; margin-left: 20px; text-align: right;">
                       <li style="font-size: 14px;">My computer is running slow. How can I speed it up? 🖥️</li>
                       <li style="font-size: 14px;">I'm getting an error message when I try to install software. What should I do? 💻</li>
                       <li style="font-size: 14px;">How do I connect my printer wirelessly to my computer? 🖨️</li>
                   </ul>
                   </div>''',
)


if __name__ == "__main__":
    demo.launch()
