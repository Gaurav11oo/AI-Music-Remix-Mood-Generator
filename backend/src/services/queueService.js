const Queue = require('bull');
const { bullRedisConfig } = require('../config/redis');

// Create queues
const audioProcessingQueue = new Queue('audio-processing', bullRedisConfig);
const stemSeparationQueue = new Queue('stem-separation', bullRedisConfig);
const moodClassificationQueue = new Queue('mood-classification', bullRedisConfig);
const musicGenerationQueue = new Queue('music-generation', bullRedisConfig);
const remixQueue = new Queue('remix', bullRedisConfig);

// Queue settings
const queueSettings = {
  attempts: 3,
  backoff: {
    type: 'exponential',
    delay: 2000
  },
  removeOnComplete: 100,
  removeOnFail: 50
};

// Add job to audio processing queue
const addAudioProcessingJob = async (jobType, data, options = {}) => {
  try {
    const job = await audioProcessingQueue.add(jobType, data, {
      ...queueSettings,
      ...options
    });
    
    console.log(`✅ Job ${job.id} added to audio-processing queue:`, jobType);
    return job;
  } catch (error) {
    console.error('Error adding job to queue:', error);
    throw error;
  }
};

// Add stem separation job
const addStemSeparationJob = async (data, options = {}) => {
  try {
    const job = await stemSeparationQueue.add('separate-stems', data, {
      ...queueSettings,
      timeout: 600000, // 10 minutes
      ...options
    });
    
    console.log(`✅ Stem separation job ${job.id} added to queue`);
    return job;
  } catch (error) {
    console.error('Error adding stem separation job:', error);
    throw error;
  }
};

// Add mood classification job
const addMoodClassificationJob = async (data, options = {}) => {
  try {
    const job = await moodClassificationQueue.add('classify-mood', data, {
      ...queueSettings,
      ...options
    });
    
    console.log(`✅ Mood classification job ${job.id} added to queue`);
    return job;
  } catch (error) {
    console.error('Error adding mood classification job:', error);
    throw error;
  }
};

// Add music generation job
const addMusicGenerationJob = async (data, options = {}) => {
  try {
    const job = await musicGenerationQueue.add('generate-music', data, {
      ...queueSettings,
      timeout: 900000, // 15 minutes
      ...options
    });
    
    console.log(`✅ Music generation job ${job.id} added to queue`);
    return job;
  } catch (error) {
    console.error('Error adding music generation job:', error);
    throw error;
  }
};

// Add remix job
const addRemixJob = async (jobType, data, options = {}) => {
  try {
    const job = await remixQueue.add(jobType, data, {
      ...queueSettings,
      ...options
    });
    
    console.log(`✅ Remix job ${job.id} (${jobType}) added to queue`);
    return job;
  } catch (error) {
    console.error('Error adding remix job:', error);
    throw error;
  }
};

// Get job status
const getJobStatus = async (queueName, jobId) => {
  try {
    let queue;
    switch (queueName) {
      case 'stem-separation':
        queue = stemSeparationQueue;
        break;
      case 'mood-classification':
        queue = moodClassificationQueue;
        break;
      case 'music-generation':
        queue = musicGenerationQueue;
        break;
      case 'remix':
        queue = remixQueue;
        break;
      default:
        queue = audioProcessingQueue;
    }

    const job = await queue.getJob(jobId);
    
    if (!job) {
      return null;
    }

    const state = await job.getState();
    const progress = job.progress();
    const result = job.returnvalue;
    const failedReason = job.failedReason;

    return {
      id: job.id,
      state,
      progress,
      result,
      failedReason,
      data: job.data,
      finishedOn: job.finishedOn,
      processedOn: job.processedOn
    };
  } catch (error) {
    console.error('Error getting job status:', error);
    throw error;
  }
};

// Event listeners for monitoring
const setupQueueMonitoring = (queue, queueName) => {
  queue.on('completed', (job, result) => {
    console.log(`✅ ${queueName} job ${job.id} completed`);
  });

  queue.on('failed', (job, err) => {
    console.error(`❌ ${queueName} job ${job.id} failed:`, err.message);
  });

  queue.on('progress', (job, progress) => {
    console.log(`⏳ ${queueName} job ${job.id} progress: ${progress}%`);
  });

  queue.on('stalled', (job) => {
    console.warn(`⚠️ ${queueName} job ${job.id} stalled`);
  });
};

// Setup monitoring for all queues
setupQueueMonitoring(audioProcessingQueue, 'audio-processing');
setupQueueMonitoring(stemSeparationQueue, 'stem-separation');
setupQueueMonitoring(moodClassificationQueue, 'mood-classification');
setupQueueMonitoring(musicGenerationQueue, 'music-generation');
setupQueueMonitoring(remixQueue, 'remix');

module.exports = {
  audioProcessingQueue,
  stemSeparationQueue,
  moodClassificationQueue,
  musicGenerationQueue,
  remixQueue,
  addAudioProcessingJob,
  addStemSeparationJob,
  addMoodClassificationJob,
  addMusicGenerationJob,
  addRemixJob,
  getJobStatus
};
