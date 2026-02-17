const express = require('express');
const { authMiddleware } = require('../middleware/auth');
const { asyncHandler } = require('../middleware/errorHandler');
const { ProcessingJob } = require('../models');
const { addMusicGenerationJob } = require('../services/queueService');

const router = express.Router();

// Generate music from text
router.post('/text-to-music', authMiddleware, asyncHandler(async (req, res) => {
  const { prompt, duration = 10, temperature = 1.0, top_k = 250 } = req.body;

  if (!prompt) {
    return res.status(400).json({
      success: false,
      message: 'Prompt is required'
    });
  }

  const job = await addMusicGenerationJob({
    prompt,
    duration,
    temperature,
    top_k,
    userId: req.userId
  });

  const processingJob = await ProcessingJob.create({
    job_id: job.id.toString(),
    user_id: req.userId,
    job_type: 'music_generation',
    status: 'pending',
    parameters: { prompt, duration, temperature, top_k }
  });

  res.status(202).json({
    success: true,
    message: 'Music generation job queued',
    data: {
      jobId: job.id,
      status: 'pending'
    }
  });
}));

// Get generation status
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

// Download generated music
router.get('/:jobId/download', authMiddleware, asyncHandler(async (req, res) => {
  const processingJob = await ProcessingJob.findOne({
    where: {
      job_id: req.params.jobId,
      user_id: req.userId,
      status: 'completed'
    }
  });

  if (!processingJob || !processingJob.result_path) {
    return res.status(404).json({
      success: false,
      message: 'Generated file not found'
    });
  }

  res.download(processingJob.result_path, 'generated-music.wav');
}));

module.exports = router;
