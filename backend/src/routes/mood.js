const express = require('express');
const { authMiddleware } = require('../middleware/auth');
const { asyncHandler } = require('../middleware/errorHandler');
const { AudioFile, MoodClassification, ProcessingJob } = require('../models');
const { addMoodClassificationJob } = require('../services/queueService');

const router = express.Router();

// Classify mood
router.post('/classify', authMiddleware, asyncHandler(async (req, res) => {
  const { audioFileId } = req.body;

  const audioFile = await AudioFile.findOne({
    where: { id: audioFileId, user_id: req.userId }
  });

  if (!audioFile) {
    return res.status(404).json({
      success: false,
      message: 'Audio file not found'
    });
  }

  const job = await addMoodClassificationJob({
    audioFileId,
    audioPath: audioFile.file_path,
    userId: req.userId
  });

  const processingJob = await ProcessingJob.create({
    job_id: job.id.toString(),
    user_id: req.userId,
    audio_file_id: audioFileId,
    job_type: 'mood_classification',
    status: 'pending'
  });

  res.status(202).json({
    success: true,
    message: 'Mood classification job queued',
    data: {
      jobId: job.id,
      status: 'pending'
    }
  });
}));

// Get mood classification
router.get('/:audioId', authMiddleware, asyncHandler(async (req, res) => {
  const audioFile = await AudioFile.findOne({
    where: { id: req.params.audioId, user_id: req.userId }
  });

  if (!audioFile) {
    return res.status(404).json({
      success: false,
      message: 'Audio file not found'
    });
  }

  const moodClassification = await MoodClassification.findOne({
    where: { audio_file_id: req.params.audioId },
    order: [['created_at', 'DESC']]
  });

  if (!moodClassification) {
    return res.status(404).json({
      success: false,
      message: 'No mood classification found for this audio file'
    });
  }

  res.json({
    success: true,
    data: moodClassification.toJSON()
  });
}));

module.exports = router;
