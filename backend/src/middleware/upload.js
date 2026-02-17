const multer = require('multer');
const path = require('path');
const fs = require('fs').promises;
const { v4: uuidv4 } = require('uuid');

// Ensure upload directories exist
const ensureDirectories = async () => {
  const dirs = [
    process.env.UPLOAD_DIR || './uploads',
    process.env.TEMP_DIR || './temp',
    path.join(process.env.UPLOAD_DIR || './uploads', 'audio'),
    path.join(process.env.UPLOAD_DIR || './uploads', 'stems'),
    path.join(process.env.UPLOAD_DIR || './uploads', 'remixes'),
    path.join(process.env.UPLOAD_DIR || './uploads', 'generated')
  ];

  for (const dir of dirs) {
    try {
      await fs.mkdir(dir, { recursive: true });
    } catch (error) {
      console.error(`Error creating directory ${dir}:`, error);
    }
  }
};

// Initialize directories
ensureDirectories();

// Configure storage
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = path.join(process.env.UPLOAD_DIR || './uploads', 'audio');
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = `${uuidv4()}-${Date.now()}`;
    const ext = path.extname(file.originalname);
    cb(null, `${uniqueSuffix}${ext}`);
  }
});

// File filter for audio files
const fileFilter = (req, file, cb) => {
  const allowedFormats = (process.env.ALLOWED_AUDIO_FORMATS || 'mp3,wav,flac,ogg,m4a,aac')
    .split(',')
    .map(f => f.trim().toLowerCase());

  const ext = path.extname(file.originalname).substring(1).toLowerCase();
  
  if (allowedFormats.includes(ext)) {
    cb(null, true);
  } else {
    cb(new Error(`Invalid file type. Allowed formats: ${allowedFormats.join(', ')}`), false);
  }
};

// Create multer upload middleware
const upload = multer({
  storage: storage,
  limits: {
    fileSize: parseInt(process.env.MAX_FILE_SIZE) || 100 * 1024 * 1024, // Default 100MB
    files: 1
  },
  fileFilter: fileFilter
});

// Upload error handler
const handleUploadError = (err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        success: false,
        message: 'File is too large',
        maxSize: process.env.MAX_FILE_SIZE
      });
    }
    if (err.code === 'LIMIT_FILE_COUNT') {
      return res.status(400).json({
        success: false,
        message: 'Too many files uploaded'
      });
    }
    return res.status(400).json({
      success: false,
      message: err.message
    });
  }

  if (err) {
    return res.status(400).json({
      success: false,
      message: err.message
    });
  }

  next();
};

module.exports = {
  upload,
  handleUploadError,
  ensureDirectories
};
