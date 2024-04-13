# music-hub

Music recomendation using face emotion

## Running rasa chatbot service for first time with training

```bash
cd rasa
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
rasa train
rasa run -m models --enable-api
```

## Running rasa chatbot service for flask

```bash
cd rasa
.venv\Scripts\activate
rasa run -m models --enable-api
```

## Flask app running

In the project root directory

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
flask run --debug
```

Go to: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
