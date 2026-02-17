const express = require('express');
const { authMiddleware } = require('../middleware/auth');
const { asyncHandler } = require('../middleware/errorHandler');
const { AudioFile, ProcessingJob } = require('../models');
const { addRemixJob } = require('../services/queueService');

const router = express.Router();

// Remix to different genre
router.post('/genre', authMiddleware, asyncHandler(async (req, res) => {
  const { audioFileId, targetGenre, intensity = 0.8 } = req.body;

  const audioFile = await AudioFile.findOne({
    where: { id: audioFileId, user_id: req.userId }
  });

  if (!audioFile) {
    return res.status(404).json({
      success: false,
      message: 'Audio file not found'
    });
  }

  const job = await addRemixJob('genre-remix', {
    audioFileId,
    audioPath: audioFile.file_path,
    targetGenre,
    intensity,
    userId: req.userId
  });

  const processingJob = await ProcessingJob.create({
    job_id: job.id.toString(),
    user_id: req.userId,
    audio_file_id: audioFileId,
    job_type: 'genre_remix',
    status: 'pending',
    parameters: { targetGenre, intensity }
  });

  res.status(202).json({
    success: true,
    message: 'Genre remix job queued',
    data: { jobId: job.id, status: 'pending' }
  });
}));

// Change tempo
router.post('/tempo', authMiddleware, asyncHandler(async (req, res) => {
  const { audioFileId, tempoChange, preservePitch = true } = req.body;

  const audioFile = await AudioFile.findOne({
    where: { id: audioFileId, user_id: req.userId }
  });

  if (!audioFile) {
    return res.status(404).json({
      success: false,
      message: 'Audio file not found'
    });
  }

  const job = await addRemixJob('tempo-change', {
    audioFileId,
    audioPath: audioFile.file_path,
    tempoChange,
    preservePitch,
    userId: req.userId
  });

  const processingJob = await ProcessingJob.create({
    job_id: job.id.toString(),
    user_id: req.userId,
    audio_file_id: audioFileId,
    job_type: 'tempo_change',
    status: 'pending',
    parameters: { tempoChange, preservePitch }
  });

  res.status(202).json({
    success: true,
    message: 'Tempo change job queued',
    data: { jobId: job.id, status: 'pending' }
  });
}));

// Change pitch
router.post('/pitch', authMiddleware, asyncHandler(async (req, res) => {
  const { audioFileId, pitchChange } = req.body;

  const audioFile = await AudioFile.findOne({
    where: { id: audioFileId, user_id: req.userId }
  });

  if (!audioFile) {
    return res.status(404).json({
      success: false,
      message: 'Audio file not found'
    });
  }

  const job = await addRemixJob('pitch-change', {
    audioFileId,
    audioPath: audioFile.file_path,
    pitchChange,
    userId: req.userId
  });

  const processingJob = await ProcessingJob.create({
    job_id: job.id.toString(),
    user_id: req.userId,
    audio_file_id: audioFileId,
    job_type: 'pitch_change',
    status: 'pending',
    parameters: { pitchChange }
  });

  res.status(202).json({
    success: true,
    message: 'Pitch change job queued',
    data: { jobId: job.id, status: 'pending' }
  });
}));

// Get remix status
router.get('/:jobId/status', authMiddleware, asyncHandler(async (req, res) => {
  const processingJob = await ProcessingJob.findOne({
    where: {
      job_id: req.params.jobId,
      user_id: req.userId
    }
  });

  if (!processingJob) {
    return res.status(404).json({
      success: false,
      message: 'Job not found'
    });
  }

  res.json({
    success: true,
    data: {
      jobId: req.params.jobId,
      status: processingJob.status,
      progress: processingJob.progress,
      result: processingJob.result_data,
      error: processingJob.error_message
    }
  });
}));

module.exports = router;
