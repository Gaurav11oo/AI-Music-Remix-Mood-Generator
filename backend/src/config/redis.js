const { createClient } = require('redis');
require('dotenv').config();

/*
|--------------------------------------------------------------------------
| Redis Configuration
|--------------------------------------------------------------------------
*/

const redisConfig = {
  host: process.env.REDIS_HOST || '127.0.0.1', // ✅ FIXED
  port: parseInt(process.env.REDIS_PORT) || 6379,
  password: process.env.REDIS_PASSWORD || undefined,
};

/*
|--------------------------------------------------------------------------
| Create Redis Client
|--------------------------------------------------------------------------
*/

const redisClient = createClient({
  socket: {
    host: redisConfig.host,
    port: redisConfig.port,
    reconnectStrategy: (retries) => {
      console.log(`🔁 Redis reconnect attempt ${retries}`);
      return Math.min(retries * 200, 3000);
    },
  },
  password: redisConfig.password,
});

/*
|--------------------------------------------------------------------------
| Events
|--------------------------------------------------------------------------
*/

redisClient.on('connect', () => {
  console.log('🟢 Redis connecting...');
});

redisClient.on('ready', () => {
  console.log('✅ Redis client ready');
});

redisClient.on('error', (err) => {
  console.error('❌ Redis Client Error:', err.message);
});

redisClient.on('end', () => {
  console.log('🔴 Redis connection closed');
});

/*
|--------------------------------------------------------------------------
| Initialize Redis
|--------------------------------------------------------------------------
*/

const initRedis = async () => {
  try {
    await redisClient.connect();
    return true;
  } catch (error) {
    console.error('❌ Redis connection failed:', error.message);
    return false;
  }
};

/*
|--------------------------------------------------------------------------
| Bull Queue Config
|--------------------------------------------------------------------------
*/

const bullRedisConfig = {
  connection: {
    host: redisConfig.host,
    port: redisConfig.port,
    password: redisConfig.password,
  },
};

module.exports = {
  redisClient,
  initRedis,
  bullRedisConfig,
  redisConfig,
};
