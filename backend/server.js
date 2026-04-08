// server.js
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const userRoutes = require('./routes/user');
require('dotenv').config();

const app = express();

// Debugging environment variables
console.log('MongoDB URI:', process.env.MONGO_URI);
console.log('PORT:', process.env.PORT);

// Connect to MongoDB
mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log('MongoDB connected'))
  .catch((err) => console.log('MongoDB connection error:', err));

// Middleware
app.use(express.json());

// Set up less restrictive CORS middleware
app.use(cors());

// User routes
app.use('/api/users', userRoutes);

// Start the server
const PORT = process.env.PORT || 5001;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
