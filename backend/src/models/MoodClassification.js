const { DataTypes } = require('sequelize');
const { sequelize } = require('../config/database');

const MoodClassification = sequelize.define('MoodClassification', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  audio_file_id: {
    type: DataTypes.INTEGER,
    allowNull: false,
    references: {
      model: 'audio_files',
      key: 'id'
    },
    onDelete: 'CASCADE'
  },
  mood: {
    type: DataTypes.STRING(50),
    allowNull: false,
    comment: 'Detected mood: happy, sad, energetic, calm, angry, romantic, etc.'
  },
  confidence: {
    type: DataTypes.FLOAT,
    allowNull: false,
    validate: {
      min: 0,
      max: 1
    }
  },
  mood_scores: {
    type: DataTypes.JSONB,
    allowNull: true,
    comment: 'Scores for all mood categories'
  },
  features: {
    type: DataTypes.JSONB,
    allowNull: true,
    comment: 'Extracted audio features (tempo, energy, valence, etc.)'
  },
  created_at: {
    type: DataTypes.DATE,
    defaultValue: DataTypes.NOW
  }
}, {
  tableName: 'mood_classifications',
  timestamps: true,
  underscored: true,
  updatedAt: false,
  indexes: [
    {
      fields: ['audio_file_id']
    },
    {
      fields: ['mood']
    }
  ]
});

module.exports = MoodClassification;
