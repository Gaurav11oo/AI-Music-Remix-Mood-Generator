const { DataTypes } = require('sequelize');
const { sequelize } = require('../config/database');

const AudioFile = sequelize.define('AudioFile', {
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
  filename: {
    type: DataTypes.STRING(255),
    allowNull: false
  },
  original_name: {
    type: DataTypes.STRING(255),
    allowNull: false
  },
  file_path: {
    type: DataTypes.STRING(500),
    allowNull: false
  },
  file_size: {
    type: DataTypes.INTEGER,
    allowNull: false,
    comment: 'File size in bytes'
  },
  duration: {
    type: DataTypes.FLOAT,
    allowNull: true,
    comment: 'Duration in seconds'
  },
  format: {
    type: DataTypes.STRING(20),
    allowNull: true,
    comment: 'Audio format (mp3, wav, etc.)'
  },
  sample_rate: {
    type: DataTypes.INTEGER,
    allowNull: true,
    comment: 'Sample rate in Hz'
  },
  channels: {
    type: DataTypes.INTEGER,
    allowNull: true,
    comment: 'Number of audio channels'
  },
  bitrate: {
    type: DataTypes.INTEGER,
    allowNull: true,
    comment: 'Bitrate in kbps'
  },
  metadata: {
    type: DataTypes.JSONB,
    allowNull: true,
    defaultValue: {},
    comment: 'Additional metadata (title, artist, album, etc.)'
  },
  created_at: {
    type: DataTypes.DATE,
    defaultValue: DataTypes.NOW
  }
}, {
  tableName: 'audio_files',
  timestamps: true,
  underscored: true,
  updatedAt: false,
  indexes: [
    {
      fields: ['user_id']
    },
    {
      fields: ['created_at']
    }
  ]
});

module.exports = AudioFile;
