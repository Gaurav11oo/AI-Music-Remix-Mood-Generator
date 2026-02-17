const User = require('./User');
const AudioFile = require('./AudioFile');
const ProcessingJob = require('./ProcessingJob');
const MoodClassification = require('./MoodClassification');
const Remix = require('./Remix');

// Define associations

// User associations
User.hasMany(AudioFile, {
  foreignKey: 'user_id',
  as: 'audioFiles'
});

User.hasMany(ProcessingJob, {
  foreignKey: 'user_id',
  as: 'jobs'
});

User.hasMany(Remix, {
  foreignKey: 'user_id',
  as: 'remixes'
});

// AudioFile associations
AudioFile.belongsTo(User, {
  foreignKey: 'user_id',
  as: 'user'
});

AudioFile.hasMany(ProcessingJob, {
  foreignKey: 'audio_file_id',
  as: 'jobs'
});

AudioFile.hasMany(MoodClassification, {
  foreignKey: 'audio_file_id',
  as: 'moodClassifications'
});

AudioFile.hasMany(Remix, {
  foreignKey: 'original_file_id',
  as: 'remixesAsOriginal'
});

AudioFile.hasMany(Remix, {
  foreignKey: 'remix_file_id',
  as: 'remixesAsRemix'
});

// ProcessingJob associations
ProcessingJob.belongsTo(User, {
  foreignKey: 'user_id',
  as: 'user'
});

ProcessingJob.belongsTo(AudioFile, {
  foreignKey: 'audio_file_id',
  as: 'audioFile'
});

// MoodClassification associations
MoodClassification.belongsTo(AudioFile, {
  foreignKey: 'audio_file_id',
  as: 'audioFile'
});

// Remix associations
Remix.belongsTo(User, {
  foreignKey: 'user_id',
  as: 'user'
});

Remix.belongsTo(AudioFile, {
  foreignKey: 'original_file_id',
  as: 'originalFile'
});

Remix.belongsTo(AudioFile, {
  foreignKey: 'remix_file_id',
  as: 'remixFile'
});

module.exports = {
  User,
  AudioFile,
  ProcessingJob,
  MoodClassification,
  Remix
};
