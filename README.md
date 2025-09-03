# MyAIFactChecker

An open-source multilingual AI-powered fact-checking platform that helps users verify news claims and combat misinformation across multiple languages including English, Hausa, Yoruba, Igbo, Swahili, French, and Arabic.

## Features

- **Multilingual Fact-Checking**: Support for 7+ languages with automatic translation
- **AI-Powered Verification**: Uses GPT-4, Groq, and multiple search APIs for comprehensive fact-checking
- **Source Tracking**: Identifies and categorizes genuine vs non-authentic sources
- **Sentiment Analysis**: Analyzes emotional tone of news claims
- **REST API**: Complete Django REST Framework API for integration
- **Export Capabilities**: CSV export for research and analysis
- **Web Interface**: User-friendly web interface for each supported language

## Tech Stack

- **Backend**: Django, Django REST Framework
- **AI/ML**: LangChain, OpenAI GPT-4, Groq, Tavily Search, Google Serper
- **NLP**: NLTK, Sentiment Analysis
- **Database**: PostgreSQL/SQLite
- **Translation**: Custom LLM-based translation system

## API Endpoints

### Core Endpoints
- `GET/POST /api/v1/factchecks/` - List and create fact-checks
- `GET /api/v1/factchecks/{slug}/` - Retrieve specific fact-check
- `POST /api/v1/fact-check/` - Single language fact-checking
- `POST /api/v1/multi-language-fact-check/` - Multi-language fact-checking
- `GET /api/v1/factchecks/export_csv/` - Export data as CSV

### Management Endpoints
- `GET/POST /api/v1/user-reports/` - User feedback and reports
- `GET /api/v1/active-users/` - Active user tracking

## Installation

### Prerequisites
- Python 3.8+
- Django 4.0+
- Node.js (for frontend assets)
- PostgreSQL (recommended) or SQLite

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/myaifactchecker.git
   cd myaifactchecker
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   
   # Database (optional)
   DATABASE_URL=postgresql://username:password@localhost:5432/myaifactchecker
   
   # Django Settings
   SECRET_KEY=your_secret_key_here
   DEBUG=True
   ALLOWED_HOSTS=localhost,127.0.0.1
   ```

5. **Database Setup**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   python manage.py collectstatic
   ```

6. **Create Superuser (optional)**
   ```bash
   python manage.py createsuperuser
   ```

7. **Run the Development Server**
   ```bash
   python manage.py runserver
   ```

## API Usage Examples

### Basic Fact-Check
```python
import requests

# Single language fact-check
response = requests.post('http://localhost:8000/api/v1/fact-check/', {
    'user_input_news': 'The Earth is flat according to recent studies'
})

# Multi-language fact-check
response = requests.post('http://localhost:8000/api/v1/multi-language-fact-check/', {
    'user_input_news': 'Duniya mai girma ne bisa ga binciken da aka yi kwanan nan',
    'language': 'hausa'
})
```

### JavaScript/Fetch Example
```javascript
// Fact-check in English
const response = await fetch('/api/v1/fact-check/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken')  // If using CSRF
    },
    body: JSON.stringify({
        user_input_news: 'Climate change is a hoax'
    })
});

const result = await response.json();
console.log(result);
```

### Response Format
```json
{
    "id": 123,
    "user_input_news": "The Earth is flat according to recent studies",
    "fresult": "This claim is false. Scientific consensus and overwhelming evidence support that the Earth is spherical...",
    "sentiment_label": "Neutral",
    "num_genuine_sources": 5,
    "non_authentic_sources": 0,
    "genuine_urls": ["https://nasa.gov/...", "https://nature.com/..."],
    "non_authentic_urls": [],
    "genuine_urls_and_dates": {},
    "slug": "the-earth-is-flat-according-to-recent-studies",
    "created_at": "2024-01-15T10:30:00Z"
}
```

## Supported Languages

- **English** - Primary language
- **Hausa** - West African lingua franca
- **Yoruba** - Nigerian language
- **Igbo** - Nigerian language  
- **Swahili** - East African lingua franca
- **French** - International language
- **Arabic** - Middle Eastern/North African

## Configuration

### Required API Keys

1. **OpenAI API Key**: For GPT-4 powered fact-checking
   - Sign up at [OpenAI](https://platform.openai.com/)
   - Add to `.env` as `OPENAI_API_KEY`

2. **Tavily API Key**: For web search and content retrieval
   - Sign up at [Tavily](https://tavily.com/)
   - Add to `.env` as `TAVILY_API_KEY`

3. **Groq API Key**: For fast LLM inference
   - Sign up at [Groq](https://groq.com/)
   - Add to `.env` as `GROQ_API_KEY`

4. **Serper API Key**: For Google Search API
   - Sign up at [Serper](https://serper.dev/)
   - Add to `.env` as `SERPER_API_KEY`

### Django Settings

Add to your `settings.py`:

```python
INSTALLED_APPS = [
    # ... existing apps
    'rest_framework',
    'corsheaders',
    'myapp',  # Your main app
]

REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.TokenAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticatedOrReadOnly',
    ],
}

# CORS settings for frontend integration
CORS_ALLOW_ALL_ORIGINS = True  # Set to False in production
```

## Development

### Running Tests
```bash
python manage.py test
```

### Code Style
```bash
# Install development dependencies
pip install black flake8 isort

# Format code
black .
isort .

# Check style
flake8 .
```

### Adding New Languages

1. Add language support in translation functions
2. Create new template files for the language
3. Update URL patterns
4. Add language option to API endpoints

## Deployment

### Production Checklist

- [ ] Set `DEBUG=False`
- [ ] Configure proper database (PostgreSQL recommended)
- [ ] Set up proper domain in `ALLOWED_HOSTS`
- [ ] Configure static files serving
- [ ] Set up SSL/HTTPS
- [ ] Configure rate limiting
- [ ] Set up monitoring and logging
- [ ] Secure API keys and environment variables

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python manage.py collectstatic --noinput

EXPOSE 8000
CMD ["gunicorn", "myproject.wsgi:application", "--bind", "0.0.0.0:8000"]
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure backward compatibility
- Test multilingual functionality

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for GPT models
- LangChain community for integration tools
- Django and DRF communities
- Contributors and maintainers

## Support

- Create an issue for bug reports or feature requests
- Join our community discussions
- Check the documentation for common questions

## Roadmap

- [ ] Add more African languages
- [ ] Implement real-time fact-checking
- [ ] Add browser extension
- [ ] Mobile app development
- [ ] Integration with social media platforms
- [ ] Advanced analytics dashboard
- [ ] Community-driven fact-checking features

## Performance

- Response time: < 3 seconds for most queries
- Supports concurrent requests
- Scalable architecture with caching
- Optimized for multilingual processing

---

**Made with ❤️ for fighting misinformation globally**
