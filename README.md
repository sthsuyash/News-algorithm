# Nepali News Algorithms

This repository contains the API and algorithms for the Nepali News project. The project aims to provide a platform for the different news machine learning algorithms that can be used in the Nepali news Portals. This also contains a minimal ui for describing the algorithms and a platform to use it, without major news portals having to implement the algorithms themselves.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

1. Clone the repository

   ```bash
   git clone <repo-url>
   ```

2. Create a virtual environment and install the dependencies

   ```bash
   python -m venv .venv
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. Run the API

   ```bash
   fastapi run app/main.py
   ```

4. Access the swagger documentation at `http://localhost:8000/docs`.

## Structure

- `app/`: API to access the news algorithms
- `news_algorithms/`: ML algorithms for the news
  - `nepali_sentiment_analysis/`: Sentiment analysis for the Nepali language
  - `news_classifier/`: News classifier for the Nepali language
  - `news_recommendation/`: News recommendation for the Nepali language
  - `text_summarizer/`: Text summarizer for the news articles
- `tests/`: Unit tests for the algorithms
- `ui/`: Minimal UI for description and usage of the algorithms

## Tests

1. Navigate to the root directory
2. Run pytest to run the tests

   ```bash
   pytest tests/
   ```

3. To run a specific test file

   ```bash
   pytest tests/test_<filename>.py
   ```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](docs/LICENSE).

- **Attribution Required**: Credit the original author (Suyash Shrestha).
- **NonCommercial**: Use of this code for commercial purposes is prohibited.
- **Modifications**: Allowed for non-commercial purposes only.

## Author

- [Suyash Shrestha](www.suyashshrestha.com.np)
