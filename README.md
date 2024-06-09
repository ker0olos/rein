# Rein

A experimental vtuber renderer using super-easy-to-make svg models.

### Roadmap

This is a (rein)carnation of a dead project I made in 2022, the plan is to
update it to use the ARkit standard 52 blendshapes instead of the current face
landmarks method, and support all those blendshapes with toggles in the model
parser.

#### How to use

Not yet ready for public use.

#### How to create a svg model

TODO

#### Dev

```sh
git clone https://github.com/ker0olos/rein && cd rein
python3 -m venv .venv && source .venv/bin/activate
pip install -e . && pip install -r requirements.txt
sh download_ml_models.sh
python3 preview_webcam.py
```
