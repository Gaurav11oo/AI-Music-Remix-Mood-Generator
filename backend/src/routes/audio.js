const express = require('express');
const { authMiddleware } = require('../middleware/auth');
const { upload, handleUploadError } = require('../middleware/upload');
const { asyncHandler } = require('../middleware/errorHandler');
const { AudioFile } = require('../models');
const fs = require('fs').promises;
const path = require('path');

const router = express.Router();

/*
|--------------------------------------------------------------------------
| Upload Audio File
|--------------------------------------------------------------------------
*/
router.post(
  '/upload',
  authMiddleware,
  upload.single('audio'),
  handleUploadError,
  asyncHandler(async (req, res) => {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'No audio file provided'
      });
    }

    try {
      // ✅ FIX: Dynamic import for ESM package
      const mm = await import('music-metadata');

      // Extract metadata
      const metadata = await mm.parseFile(req.file.path);

      // Save into DB
      const audioFile = await AudioFile.create({
        user_id: req.userId,
        filename: req.file.filename,
        original_name: req.file.originalname,
        file_path: req.file.path,
        file_size: req.file.size,

        duration: metadata?.format?.duration || null,
        format: metadata?.format?.container || null,
        sample_rate: metadata?.format?.sampleRate || null,
        channels: metadata?.format?.numberOfChannels || null,
        bitrate: metadata?.format?.bitrate || null,

        metadata: {
          title: metadata?.common?.title || null,
          artist: metadata?.common?.artist || null,
          album: metadata?.common?.album || null,
          year: metadata?.common?.year || null
        }
      });

      return res.status(201).json({
        success: true,
        message: 'Audio uploaded successfully',
        data: {
          audioFile: audioFile.toJSON()
        }
      });

    } catch (error) {
      // delete uploaded file if error occurs
      await fs.unlink(req.file.path).catch(() => {});
      throw error;
    }
  })
);

/*
|--------------------------------------------------------------------------
| List User Audio Files
|--------------------------------------------------------------------------
*/
router.get(
  '/',
  authMiddleware,
  asyncHandler(async (req, res) => {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const offset = (page - 1) * limit;

    const { count, rows } = await AudioFile.findAndCountAll({
      where: { user_id: req.userId },
      limit,
      offset,
      order: [['created_at', 'DESC']]
    });

    res.json({
      success: true,
      data: {
        audioFiles: rows,
        pagination: {
          page,
          limit,
          total: count,
          pages: Math.ceil(count / limit)
        }
      }
    });
  })
);

/*
|--------------------------------------------------------------------------
| Get Single Audio File
|--------------------------------------------------------------------------
*/
router.get(
  '/:id',
  authMiddleware,
  asyncHandler(async (req, res) => {
    const audioFile = await AudioFile.findOne({
      where: {
        id: req.params.id,
        user_id: req.userId
      }
    });

    if (!audioFile) {
      return res.status(404).json({
        success: false,
        message: 'Audio file not found'
      });
    }

    res.json({
      success: true,
      data: {
        audioFile: audioFile.toJSON()
      }
    });
  })
);

/*
|--------------------------------------------------------------------------
| Download Audio File
|--------------------------------------------------------------------------
*/
router.get(
  '/:id/download',
  authMiddleware,
  asyncHandler(async (req, res) => {
    const audioFile = await AudioFile.findOne({
      where: {
        id: req.params.id,
        user_id: req.userId
      }
    });

    if (!audioFile) {
      return res.status(404).json({
        success: false,
        message: 'Audio file not found'
      });
    }

    res.download(
      path.resolve(audioFile.file_path),
      audioFile.original_name
    );
  })
);

/*
|--------------------------------------------------------------------------
| Delete Audio File
|--------------------------------------------------------------------------
*/
router.delete(
  '/:id',
  authMiddleware,
  asyncHandler(async (req, res) => {
    const audioFile = await AudioFile.findOne({
      where: {
        id: req.params.id,
        user_id: req.userId
      }
    });

    if (!audioFile) {
      return res.status(404).json({
        success: false,
        message: 'Audio file not found'
      });
    }

    // Delete physical file
    try {
      await fs.unlink(audioFile.file_path);
    } catch (err) {
      console.error('File delete error:', err.message);
    }

    // Delete DB record
    await audioFile.destroy();

    res.json({
      success: true,
      message: 'Audio file deleted successfully'
    });
  })
);

module.exports = router;
