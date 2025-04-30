const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const User = require('../models/User');
const router = express.Router();
const authMiddleware = require('../middleware/auth');
require('dotenv').config();

// Explicitly handle CORS preflight requests
router.options('*', (req, res) => {
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, x-auth-token');
  res.sendStatus(200);
});

// Register route with enhanced logging
router.post('/register', async (req, res) => {
  const { name, email, password, phone } = req.body;

  try {
    console.log(`Attempting to register user with email: ${email}`);

    // Create new user (password will be hashed by pre-save hook)
    const user = new User({ name, email, password, phone });
    await user.save();

    console.log(`User registered successfully with email: ${email}`);
    res.status(201).send('User registered successfully');
  } catch (error) {
    console.error('Error registering user:', error);
    res.status(500).send('Error registering user. Please try again.');
  }
});

// Login route with improved password comparison logic
router.post('/login', async (req, res) => {
  const { email, password } = req.body;

  try {
    console.log(`Attempting to find user with email: ${email}`);
    const user = await User.findOne({ email });
    if (!user) {
      console.error('User not found with provided email.');
      return res.status(400).send('Invalid email or password.');
    }

    // Use the comparePassword method from the User model
    console.log(`Comparing password for user: ${user.email}`);
    const isMatch = await user.comparePassword(password);
    if (!isMatch) {
      console.error('Password does not match.');
      return res.status(400).send('Invalid email or password.');
    }

    console.log(`Generating token for user: ${user.email}`);
    const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET, { expiresIn: '1h' });
    res.json({ token });
  } catch (error) {
    console.error('Error logging in:', error);
    res.status(500).send('Error logging in. Please try again.');
  }
});

// Get account info (protected)
router.get('/account', authMiddleware, async (req, res) => {
  try {
    const user = await User.findById(req.user.userId).select('-password');
    if (!user) {
      return res.status(404).send('User not found.');
    }
    res.json(user);
  } catch (error) {
    console.error('Error fetching account info:', error);
    res.status(500).send('Server error. Please try again.');
  }
});

// Buy Cryptocurrency
router.post('/buy', authMiddleware, async (req, res) => {
  const { amount, selectedCrypto, price } = req.body;

  try {
    const user = await User.findById(req.user.userId);
    if (!user) {
      return res.status(404).send('User not found.');
    }

    const totalCost = amount * price;

    let actualAmountBought = amount;
    if (user.balance < totalCost) {
      // Calculate the fractional amount that can be bought with the available balance
      actualAmountBought = user.balance / price;
    }

    const finalCost = actualAmountBought * price;

    // Update user balance and add transaction to statements
    user.balance -= finalCost;
    user.statements.push(
      `Bought ${actualAmountBought.toFixed(6)} of ${selectedCrypto} for ${finalCost.toFixed(2)} USD`
    );
    await user.save();

    res.send(`Purchase successful. Bought ${actualAmountBought.toFixed(6)} of ${selectedCrypto}.`);
  } catch (error) {
    console.error('Error buying cryptocurrency:', error);
    res.status(500).send('Server error. Please try again.');
  }
});

// Sell Cryptocurrency
router.post('/sell', authMiddleware, async (req, res) => {
  const { amount, selectedCrypto, price } = req.body;

  try {
    const user = await User.findById(req.user.userId);
    if (!user) {
      return res.status(404).send('User not found.');
    }

    const totalValue = amount * price;

    // Update user balance and add transaction to statements
    user.balance += totalValue;
    user.statements.push(
      `Sold ${amount.toFixed(6)} of ${selectedCrypto} for ${totalValue.toFixed(2)} USD`
    );
    await user.save();

    res.send(`Sell successful. Sold ${amount.toFixed(6)} of ${selectedCrypto}.`);
  } catch (error) {
    console.error('Error selling cryptocurrency:', error);
    res.status(500).send('Server error. Please try again.');
  }
});


module.exports = router;
