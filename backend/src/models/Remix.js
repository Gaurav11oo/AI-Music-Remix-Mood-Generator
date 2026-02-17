const { DataTypes } = require('sequelize');
const { sequelize } = require('../config/database');

const Remix = sequelize.define('Remix', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  user_id: {
    type: DataTypes.INTEGER,
    allowNull: false,
    references: {
      model: 'users',
      key: 'id'
    },
    onDelete: 'CASCADE'
  },
  original_file_id: {
    type: DataTypes.INTEGER,
    allowNull: true,
    references: {
      model: 'audio_files',
      key: 'id'
    },
    onDelete: 'SET NULL'
  },
  remix_file_id: {
    type: DataTypes.INTEGER,
    allowNull: false,
    references: {
      model: 'audio_files',
      key: 'id'
    },
    onDelete: 'CASCADE'
  },
  genre: {
    type: DataTypes.STRING(50),
    allowNull: true,
    comment: 'Target genre for remix'
  },
  tempo_change: {
    type: DataTypes.FLOAT,
    allowNull: true,
    comment: 'Tempo change factor (1.0 = no change, 1.5 = 50% faster)'
  },
  pitch_change: {
    type: DataTypes.FLOAT,
    allowNull: true,
    comment: 'Pitch change in semitones'
  },
  effects: {
    type: DataTypes.JSONB,
    allowNull: true,
    defaultValue: {},
    comment: 'Applied effects and their parameters'
  },
  created_at: {
    type: DataTypes.DATE,
    defaultValue: DataTypes.NOW
  }
}, {
  tableName: 'remixes',
  timestamps: true,
  underscored: true,
  updatedAt: false,
  indexes: [
    {
      fields: ['user_id']
    },
    {
      fields: ['original_file_id']
    }
  ]
});

module.exports = Remix;
