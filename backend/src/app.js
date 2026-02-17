const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const path = require('path');
require('dotenv').config();

// Import configurations
const { testConnection, syncDatabase } = require('./config/database');
const { initRedis } = require('./config/redis');

// Import middleware
const { notFound, errorHandler } = require('./middleware/errorHandler');

// Import routes
const authRoutes = require('./routes/auth');
const audioRoutes = require('./routes/audio');
const stemRoutes = require('./routes/stems');
const moodRoutes = require('./routes/mood');
const remixRoutes = require('./routes/remix');
const generateRoutes = require('./routes/generate');

// Initialize Express app
const app = express();

// Security middleware
app.use(helmet());

// CORS configuration
app.use(cors({
  origin: process.env.CORS_ORIGIN || 'http://localhost:3000',
  credentials: true
}));

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Logging middleware
if (process.env.NODE_ENV === 'development') {
  app.use(morgan('dev'));
} else {
  app.use(morgan('combined'));
}

// Static file serving
app.use('/uploads', express.static(path.join(__dirname, '../uploads')));

// API Routes
app.use('/api/auth', authRoutes);
app.use('/api/audio', audioRoutes);
app.use('/api/stems', stemRoutes);
app.use('/api/mood', moodRoutes);
app.use('/api/remix', remixRoutes);
app.use('/api/generate', generateRoutes);

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    success: true,
    message: 'Server is running',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// API documentation (Swagger)
if (process.env.NODE_ENV === 'development') {
  const swaggerJsDoc = require('swagger-jsdoc');
  const swaggerUi = require('swagger-ui-express');

  const swaggerOptions = {
    definition: {
      openapi: '3.0.0',
      info: {
        title: 'Music Remix API',
        version: '1.0.0',
        description: 'AI-powered music remix and mood generation API'
      },
      servers: [
        {
          url: process.env.API_BASE_URL || 'http://localhost:5000',
          description: 'Development server'
        }
      ],
      components: {
        securitySchemes: {
          bearerAuth: {
            type: 'http',
            scheme: 'bearer',
            bearerFormat: 'JWT'
          }
        }
      }
    },
    apis: ['./src/routes/*.js']
  };

  const swaggerDocs = swaggerJsDoc(swaggerOptions);
  app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocs));
}

// Error handling
app.use(notFound);
app.use(errorHandler);

// Initialize server
const PORT = process.env.PORT || 5000;

const startServer = async () => {
  try {
    // Test database connection
    const dbConnected = await testConnection();
    if (!dbConnected) {
      throw new Error('Database connection failed');
    }

    // Sync database models
    await syncDatabase(false);

    // Initialize Redis
    const redisConnected = await initRedis();
    if (!redisConnected) {
      console.warn('⚠️ Redis connection failed - queue features will not work');
    }

    // Start server
    app.listen(PORT, () => {
      console.log(`
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🎵  Music Remix & Mood Generator API                    ║
║                                                            ║
║   Environment: ${process.env.NODE_ENV?.padEnd(37) || 'development'.padEnd(37)}║
║   Port: ${PORT.toString().padEnd(48)}║
║   API URL: ${(process.env.API_BASE_URL || `http://localhost:${PORT}`).padEnd(44)}║
║                                                           ║
║   📖 API Docs: http://localhost:${PORT}/api-docs        ║
║   🏥 Health: http://localhost:${PORT}/api/health         ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
      `);
    });
  } catch (error) {
    console.error('❌ Server startup failed:', error);
    process.exit(1);
  }
};

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully');
  process.exit(0);
});

// Start the server
startServer();

module.exports = app;
