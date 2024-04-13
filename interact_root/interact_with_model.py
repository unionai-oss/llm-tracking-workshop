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

# %%
"""
## Positive Reviews

"Everything Everywhere All at Once" is a mind-bending martial arts film with a
surprising amount of heart. Michelle Yeoh delivers a phenomenal performance as a
struggling laundromat owner who discovers she must connect with parallel universe
versions of herself to save the world. Buckle up for wild action sequences and a
touching exploration of family and regret.

"The Batman" takes a dark and brooding look at the iconic superhero. Robert Pattinson is
captivating as a vengeance-fueled Batman in his second year of fighting crime. The film
boasts a stunning neo-noir atmosphere and a gripping mystery that will keep you guessing
until the very end.

"Spider-Man: No Way Home" is a nostalgic and action-packed adventure. Tom Holland is
charming as ever as Peter Parker, whose world is turned upside down when he accidentally
opens the multiverse. Prepare for emotional reunions with past Spider-Men and villains,
along with jaw-dropping special effects.

"DUNE" is a visually stunning and epic science fiction film. Timothée Chalamet leads an
all-star cast in this adaptation of Frank Herbert's classic novel. Expect breathtaking
desert landscapes, giant sandworms, and political intrigue on a galactic scale.

## Negative Reviews

"A Slow Burn" promises a suspenseful thriller but delivers a tedious slog. The plot
takes forever to get going, and the characters are underdeveloped and uninteresting.
Avoid this one unless you enjoy watching paint dry.

"Cosmic Catastrophe" boasts impressive CGI effects, but the story is a mess. The
nonsensical plot holes are numerous, and the dialogue is cringe-worthy. This film is all
spectacle and no substance.

"Comedy Caper" tries way too hard to be funny, relying on slapstick humor and tired
jokes. The acting is broad and uninspired, and the predictable plot offers no surprises.
Save your money and skip this one.

"Knightmare on Ice" wastes a talented cast with a nonsensical plot. The action sequences
are poorly choreographed, and the romance feels forced. This fantasy film is a missed
opportunity on all fronts.
"""
