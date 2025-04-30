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

## Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 14.0 or higher
- npm or yarn
- Git
- A modern web browser (Chrome, Firefox, Safari, or Edge)

### Important Note About Virtual Environments

The project uses Python virtual environments to manage dependencies. These environments are local to your machine and should NOT be committed to the repository. The `requirements.txt` file contains all necessary dependencies.

⚠️ Never commit the virtual environment folders (`.venv`, `venv`, etc.) to git!

If you're having issues with large files when pushing to GitHub:
1. Ensure you have the latest `.gitignore` file
2. Remove any virtual environment from git tracking:
   ```bash
   git rm -r --cached .venv
   git rm -r --cached venv
   ```
3. Create a new virtual environment after cloning:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Unix/macOS
   # OR
   .\.venv\Scripts\activate  # On Windows
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Detailed Setup Instructions

#### 1. System Preparation

First, ensure you have all required tools installed:

```bash
# Check Python version
python --version  # Should be 3.8 or higher

# Check Node.js version
node --version   # Should be 14.0 or higher

# Check npm version
npm --version
```

#### 2. Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HERMES.git
cd HERMES
```

2. Create and activate a virtual environment:
```bash
# For Unix/macOS
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install --upgrade pip  # Ensure pip is up to date
pip install -r requirements.txt
```

4. Verify backend dependencies:
```bash
python -c "import fastapi; import uvicorn; print('Dependencies installed successfully!')"
```

5. Start the FastAPI backend:
```bash
cd backend
uvicorn App:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`

#### 3. Frontend Setup

1. Navigate to the frontend directory:
```bash
cd new-frontend
```

2. Install Node.js dependencies:
```bash
# Using npm
npm install

# OR using yarn
yarn install
```

3. Create a .env file:
```bash
echo "REACT_APP_API_URL=http://localhost:8000" > .env
```

4. Start the development server:
```bash
npm start
```

The application will be available at `http://localhost:3000`

### Troubleshooting Common Issues

1. **Backend Dependencies**
   - If you encounter SSL errors during pip install:
     ```bash
     pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
     ```
   - For Windows users experiencing build errors:
     ```bash
     pip install wheel
     pip install -r requirements.txt
     ```

2. **Frontend Dependencies**
   - If you encounter node-gyp errors:
     ```bash
     # On Windows
     npm install --global windows-build-tools
     
     # On macOS
     xcode-select --install
     ```
   - Clear npm cache if needed:
     ```bash
     npm cache clean --force
     ```

3. **Running the Application**
   - If the backend fails to start, check if port 8000 is available:
     ```bash
     # On Unix/macOS
     lsof -i :8000
     
     # On Windows
     netstat -ano | findstr :8000
     ```
   - If the frontend fails to connect, verify the backend URL in .env

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