# Medicograd FastAPI API

This is a FastAPI API for predicting topics based on transcripts using the Medicograd model. The API takes a transcript text as input and returns the predicted topic from the model.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your_username/medicograd-fastapi.git
cd medicograd-fastapi
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Download required NLTK data:

```bash
python -m nltk.downloader punkt wordnet stopwords
```

4. Place the Medicograd model file (`medicograd_V2_model.h5`) and tokenizer file (`tokenizer.json`) in the project directory.

## Running the API

To run the FastAPI server, execute the following command:

```bash
uvicorn app:app --reload
```

The API will be accessible at `http://localhost:8000`.

## Usage

Send a POST request to `http://localhost:8000/predict/` with the transcript text in the request body as follows:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "transcript": "Your transcript text here."
}'
```

The API will return the predicted topic in the response JSON.

## Error Handling

If there is an error in the input JSON format or any other internal server error, the API will return an appropriate error message with the corresponding HTTP status code.

## Model and Tokenizer

The API uses the Medicograd model (`medicograd_V2_model.h5`) for topic prediction. The tokenizer information is loaded from `tokenizer.json` to preprocess the text before feeding it to the model.

## Credits

This API was developed using FastAPI and TensorFlow/Keras. The Medicograd model was trained by [author name] and is based on [dataset information, etc.]. The NLTK library was used for natural language processing.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to use, modify, and distribute the code as per the terms of the license.

For any issues or suggestions, please create an issue on the [GitHub repository](https://github.com/your_username/medicograd-fastapi).
