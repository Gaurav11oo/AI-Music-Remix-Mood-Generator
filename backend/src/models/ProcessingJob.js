const { DataTypes } = require('sequelize');
const { sequelize } = require('../config/database');

const ProcessingJob = sequelize.define('ProcessingJob', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  job_id: {
    type: DataTypes.STRING(100),
    allowNull: false,
    unique: true,
    comment: 'Bull queue job ID'
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
  audio_file_id: {
    type: DataTypes.INTEGER,
    allowNull: true,
    references: {
      model: 'audio_files',
      key: 'id'
    },
    onDelete: 'CASCADE'
  },
  job_type: {
    type: DataTypes.STRING(50),
    allowNull: false,
    comment: 'stem_separation, mood_classification, music_generation, remix, etc.'
  },
  status: {
    type: DataTypes.STRING(50),
    defaultValue: 'pending',
    comment: 'pending, processing, completed, failed'
  },
  progress: {
    type: DataTypes.INTEGER,
    defaultValue: 0,
    validate: {
      min: 0,
      max: 100
    }
  },
  result_path: {
    type: DataTypes.STRING(500),
    allowNull: true,
    comment: 'Path to result file(s)'
  },
  result_data: {
    type: DataTypes.JSONB,
    allowNull: true,
    defaultValue: {},
    comment: 'Additional result data'
  },
  parameters: {
    type: DataTypes.JSONB,
    allowNull: true,
    defaultValue: {},
    comment: 'Job parameters'
  },
  error_message: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  created_at: {
    type: DataTypes.DATE,
    defaultValue: DataTypes.NOW
  },
  started_at: {
    type: DataTypes.DATE,
    allowNull: true
  },
  completed_at: {
    type: DataTypes.DATE,
    allowNull: true
  }
}, {
  tableName: 'processing_jobs',
  timestamps: true,
  underscored: true,
  updatedAt: false,
  indexes: [
    {
      fields: ['job_id']
    },
    {
      fields: ['user_id', 'status']
    },
    {
      fields: ['created_at']
    }
  ]
});

module.exports = ProcessingJob;
