import importlib as imp
import pickle

import chatbot
import gradio as gr
import torch


def get_latest_model(path="/data/home/ze.song/models/gpt"):

    with open(
        f"{path}/logs.pkl",
        "rb",
    ) as f:
        log = pickle.load(f)

    epoch = log[-1][0].replace("epoch : ", "")
    iter = log[-1][1].replace("iter: ", "")

    model_path = f"{path}/model_{epoch}_{iter}.pkl"
    return model_path


model_path = get_latest_model()
vocab_path = (
    "/data/home/ze.song/git/block/src/block/bricks/train/gpt/model_files/vocab.pkl"
)
config = {
    "vocab_size": 13317,
    "embd_pdrop": 0.1,
    "n_embd": 1024,
    "n_head": 16,
    "n_positions": 320,
    "n_layer": 12,
    "attn_pdrop": 0.1,
    "resid_dropout": 0.1,
    "n_inner": 1024 * 4,
    "layer_norm_epsilon": 1e-5,
    "pad_idx": 0,
    "dtype": torch.float32,
    "segment_size": 3,
}

imp.reload(chatbot)
bot = chatbot.Bot(model_path=model_path, vocab_path=vocab_path, config=config)


def predict(input):

    _ = bot.talk(input, top_k=None, top_p=0.7)
    return bot.his


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()

    with gr.Row():
        txt = gr.Textbox(
            show_label=False, placeholder="Enter text and press enter"
        ).style(container=False)

    txt.submit(predict, txt, chatbot)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", debug=True, server_port=7861)