import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize global variables
adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Read reviews from the file
reviews = pd.read_csv('data/reviews.csv').to_dict('records')

# Read locations from the file
with open('data/locations.txt', 'r') as file:
    valid_locations = [line.strip() for line in file.readlines()]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        # Analyze the sentiment of the review body using VADER
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            try:
                # Parse the query string
                parsed_query = parse_qs(environ["QUERY_STRING"])
                location = parsed_query.get("location", [None])[0]
                start_date_str = parsed_query.get("start_date", [None])[0]
                end_date_str = parsed_query.get("end_date", [None])[0]

                # Check if location is valid
                if location and location not in valid_locations:
                    raise ValueError("Invalid location")

                # Convert start_date and end_date to datetime objects
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else None
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else None

                # Filter reviews based on location and date range
                filtered_reviews = [
                    review for review in reviews
                    if (not location or review["Location"] == location) and
                    (not start_date or datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") >= start_date) and
                    (not end_date or datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") <= end_date)
                ]

                for review in filtered_reviews:
                    review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

                # Sort the filtered reviews by the compound value in sentiment in descending order
                sorted_reviews = sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse=True)

                # Create the response body from the reviews and convert to a JSON byte string
                response_body = json.dumps(sorted_reviews, indent=2).encode("utf-8")

                # Set the appropriate response headers
                start_response("200 OK", [("Content-Type", "application/json"), ("Content-Length", str(len(response_body)))])
                return [response_body]
            except Exception as e:
                # Handle errors
                start_response('400 Bad Request', [('Content-Type', 'application/json')])
                return [json.dumps({'status': 'error', 'message': str(e)}).encode('utf-8')]

        if environ["REQUEST_METHOD"] == "POST":
            try:
                # Parse the POST request body
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(request_body_size)
                params = parse_qs(request_body.decode('utf-8'))
                
                # Extract ReviewBody and Location
                review_body = params.get('ReviewBody', [''])[0]
                location = params.get('Location', [''])[0]

                # Check if location is valid
                if location not in valid_locations:
                    raise ValueError("Invalid location")
                if not location:
                    raise ValueError("Location is required")
                if not review_body:
                    raise ValueError("ReviewBody is required")
                
                # Generate Timestamp and ReviewId
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                review_id = str(uuid.uuid4())
                
                # Store the review data
                review_data = {
                    'ReviewBody': review_body,
                    'Location': location,
                    'Timestamp': timestamp,
                    'ReviewId': review_id
                }
                
                # Create the response body from the reviews and convert to a JSON byte string
                response_body = json.dumps(review_data, indent=2).encode("utf-8")

                # Set the appropriate response headers
                start_response('201 OK', [('Content-Type', 'application/json')])
                return [response_body]
            
            except Exception as e:
                # Handle errors
                start_response('400 Bad Request', [('Content-Type', 'application/json')])
                return [json.dumps({'status': 'error', 'message': str(e)}).encode('utf-8')]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()