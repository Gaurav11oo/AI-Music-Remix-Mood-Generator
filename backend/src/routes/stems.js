const express = require('express');
const { authMiddleware } = require('../middleware/auth');
const { asyncHandler } = require('../middleware/errorHandler');
const { AudioFile, ProcessingJob } = require('../models');
const { addStemSeparationJob, getJobStatus } = require('../services/queueService');

const router = express.Router();

// Separate stems
router.post('/separate', authMiddleware, asyncHandler(async (req, res) => {
  const { audioFileId, model = 'htdemucs', stems } = req.body;

  const audioFile = await AudioFile.findOne({
    where: { id: audioFileId, user_id: req.userId }
  });

  if (!audioFile) {
    return res.status(404).json({
      success: false,
      message: 'Audio file not found'
    });
  }

  const job = await addStemSeparationJob({
    audioFileId,
    audioPath: audioFile.file_path,
    model,
    stems: stems || ['vocals', 'drums', 'bass', 'other'],
    userId: req.userId
  });

  const processingJob = await ProcessingJob.create({
    job_id: job.id.toString(),
    user_id: req.userId,
    audio_file_id: audioFileId,
    job_type: 'stem_separation',
    status: 'pending',
    parameters: { model, stems }
  });

  res.status(202).json({
    success: true,
    message: 'Stem separation job queued',
    data: {
      jobId: job.id,
      status: 'pending'
    }
  });
}));

// Get job status
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

  const jobStatus = await getJobStatus('stem-separation', req.params.jobId);

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

// Download stems
router.get('/:jobId/download', authMiddleware, asyncHandler(async (req, res) => {
  const processingJob = await ProcessingJob.findOne({
    where: {
      job_id: req.params.jobId,
      user_id: req.userId,
      status: 'completed'
    }
  });

  if (!processingJob) {
    return res.status(404).json({
      success: false,
      message: 'Job not found or not completed'
    });
  }

  // Return stems paths (in real implementation, create ZIP file)
  res.json({
    success: true,
    data: {
      stems: processingJob.result_data.stems
    }
  });
}));

module.exports = router;
