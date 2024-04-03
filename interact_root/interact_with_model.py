# %%
from transformers import pipeline
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display

inference = pipeline("text-classification", model="/root/model_dir")

text_area = widgets.Textarea(
    value="", rows=6, disabled=False, layout=Layout(width="auto")
)
out = widgets.Output()
submit_button = widgets.Button(description="Submit text", button_style="success")


def on_button_clicked(b):
    out.clear_output()
    with out:
        print(inference(text_area.value))


submit_button.on_click(on_button_clicked)

display(text_area, submit_button, out)
