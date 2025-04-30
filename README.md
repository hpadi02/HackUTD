# HERMES - AI-Powered Cryptocurrency Trading Platform 🚀
#### 🏆 Winner of PNC Bank's Challenge at UTDHacks '24

## Overview

HERMES is a modern, AI-powered cryptocurrency trading platform that combines real-time market data with advanced risk assessment to help users make informed trading decisions. Built with React and FastAPI, HERMES provides a seamless and secure trading experience with features designed to empower both novice and experienced traders.

![HERMES Login](screenshots/login.png)

### Key Features

- 🔒 Secure user authentication and account management
- 💹 Real-time cryptocurrency price tracking
- 📊 Interactive historical price charts with zoom functionality
- 🎯 AI-powered risk assessment meter
- 💰 Simple buy/sell interface with real-time balance updates
- 📱 Modern, responsive user interface
- 🔄 Automatic price and portfolio updates

## Screenshots

### Crypto Dashboard
![Crypto Dashboard](screenshots/dashboard.png)
The main dashboard features real-time pricing, an innovative risk assessment meter, and historical data visualization.

## Installation and Setup

### Prerequisites
- Python 3.11 or higher
- Node.js 16 or higher
- npm (comes with Node.js)
- Git

### Backend Setup

1. Create and activate a Python virtual environment:
   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate virtual environment
   # On macOS/Linux:
   source .venv/bin/activate
   # On Windows:
   .\.venv\Scripts\activate
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   # Create a .env file in the backend directory
   touch backend/.env
   ```
   Add the following to your `.env` file:
   ```
   SECRET_KEY=your_secret_key_here
   DATABASE_URL=your_database_url_here
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd new-frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file in the frontend directory:
   ```bash
   touch .env
   ```
   Add the following to your `.env` file:
   ```
   REACT_APP_API_URL=http://localhost:8000
   ```

### Running the Application

1. Start the backend server:
   ```bash
   # From the root directory
   cd backend
   uvicorn App:app --reload
   ```

2. Start the frontend development server:
   ```bash
   # From the new-frontend directory
   npm start
   ```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

### Important Notes

1. **Virtual Environments**: 
   - Always use a virtual environment for Python development
   - Never commit the virtual environment to git
   - The `.gitignore` file is configured to exclude virtual environments and other unnecessary files

2. **Dependencies**:
   - All Python dependencies are listed in `requirements.txt`
   - All Node.js dependencies are listed in `package.json`
   - Run `pip install -r requirements.txt` and `npm install` after cloning the repository

3. **Environment Variables**:
   - Keep your `.env` files secure and never commit them to git
   - Create `.env` files from the provided templates

## Project Structure

```
HERMES/
├── backend/
│   ├── App.py           # FastAPI application
│   └── requirements.txt # Python dependencies
├── new-frontend/
│   ├── src/
│   │   ├── components/  # React components
│   │   └── styles/      # CSS styles
│   ├── package.json
│   └── README.md
└── README.md
```

## Technologies Used

### Frontend
- React.js
- Chart.js for data visualization
- Axios for API requests
- Modern CSS with Flexbox

### Backend
- FastAPI
- JWT for authentication
- Pydantic for data validation
- CORS middleware for security

## Features in Detail

### User Authentication
- Secure registration and login system
- JWT token-based authentication
- Protected API endpoints

### Cryptocurrency Trading
- Real-time price updates
- Support for multiple cryptocurrencies
- Instant buy/sell transactions
- Automatic balance updates

### Risk Assessment
- AI-powered risk scoring system
- Visual risk meter with gradient indicators
- Real-time risk updates based on market conditions
- Detailed risk explanations via tooltips

### Market Analysis
- Interactive historical price charts
- Multiple timeframe options (1M, 3M, 6M, 1Y)
- Zoom functionality for detailed analysis
- Real-time price updates

## Security Features

- JWT token authentication
- Secure password handling
- Protected API endpoints
- CORS security
- Input validation
- Error handling

## About the Project

HERMES was developed as part of UTDHacks '24 and won PNC Bank's Challenge. The project demonstrates the potential of combining traditional banking principles with modern cryptocurrency trading, featuring an AI-powered risk assessment system that helps users make informed investment decisions.

## Contributing

We welcome contributions to HERMES! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PNC Bank for the challenge opportunity at UTDHacks '24
- The UTD Hackathon organizing team
- All contributors and team members

---

For questions or support, please open an issue in the repository. 