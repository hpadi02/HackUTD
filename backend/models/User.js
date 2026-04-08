const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const UserSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, unique: true, required: true },
  password: { type: String, required: true },
  phone: { type: String, required: true },
  balance: { type: Number, default: 5000 },  // Default balance
  currency: { type: String, default: 'USD' },
  creditScore: { type: Number, default: Math.floor(Math.random() * 301) + 500 }, // Randomized between 500-800
  statements: [
    {
      type: String, // Alternatively, use an object for more detailed records.
      default: [],
    },
  ],
}, { timestamps: true });

// Pre-save hook to hash password before saving it to the database
UserSchema.pre('save', async function (next) {
  if (!this.isModified('password')) {
    return next(); // Only hash the password if it's new or modified
  }

  try {
    console.log('Hashing password before saving user...');
    const salt = await bcrypt.genSalt(10);
    this.password = await bcrypt.hash(this.password, salt);
    console.log(`Password hashed successfully: ${this.password}`);
    next();
  } catch (error) {
    console.error('Error hashing password:', error);
    return next(error);
  }
});

// Method to compare passwords for login
UserSchema.methods.comparePassword = async function (enteredPassword) {
  try {
    console.log(`Comparing entered password with stored hash...`);
    const isMatch = await bcrypt.compare(enteredPassword, this.password);
    console.log(`Password match result: ${isMatch}`);
    return isMatch;
  } catch (error) {
    console.error('Error comparing passwords:', error);
    throw new Error('Password comparison failed');
  }
};

module.exports = mongoose.model('User', UserSchema);
