from implement_rag.answer import *
import gradio as gr


def chat(question, history):
    answer, docs = answer_question(question, history)

    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})

    docs_json = [
            {
                "page_content": d.page_content,
                "metadata": d.metadata
            }
            for d in docs
        ]

    return history, "", docs_json

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question")
    sources = gr.JSON(label="Retrieved Documents")


    msg.submit(
        chat,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, sources]
    )

demo.launch()