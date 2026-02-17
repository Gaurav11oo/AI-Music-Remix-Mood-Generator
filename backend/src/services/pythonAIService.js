const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

const PYTHON_AI_URL = process.env.PYTHON_AI_URL || 'http://localhost:5001';

class PythonAIService {
  constructor() {
    this.baseURL = PYTHON_AI_URL;
    this.timeout = 600000; // 10 minutes
  }

  // Health check
  async healthCheck() {
    try {
      const response = await axios.get(`${this.baseURL}/health`, {
        timeout: 5000
      });
      return response.data;
    } catch (error) {
      console.error('Python AI service health check failed:', error.message);
      throw new Error('Python AI service is not available');
    }
  }

  // Separate stems
  async separateStems(audioPath, options = {}) {
    try {
      const formData = new FormData();
      formData.append('audio', fs.createReadStream(audioPath));
      formData.append('model', options.model || 'htdemucs');
      
      if (options.stems) {
        formData.append('stems', JSON.stringify(options.stems));
      }

      const response = await axios.post(`${this.baseURL}/separate-stems`, formData, {
        headers: formData.getHeaders(),
        timeout: this.timeout,
        maxContentLength: Infinity,
        maxBodyLength: Infinity
      });

      return response.data;
    } catch (error) {
      console.error('Stem separation error:', error.message);
      throw this.handleError(error);
    }
  }

  // Classify mood
  async classifyMood(audioPath) {
    try {
      const formData = new FormData();
      formData.append('audio', fs.createReadStream(audioPath));

      const response = await axios.post(`${this.baseURL}/classify-mood`, formData, {
        headers: formData.getHeaders(),
        timeout: 60000
      });

      return response.data;
    } catch (error) {
      console.error('Mood classification error:', error.message);
      throw this.handleError(error);
    }
  }

  // Generate music from text
  async generateMusic(prompt, options = {}) {
    try {
      const response = await axios.post(`${this.baseURL}/generate-music`, {
        prompt,
        duration: options.duration || 10,
        temperature: options.temperature || 1.0,
        top_k: options.top_k || 250,
        top_p: options.top_p || 0.0
      }, {
        timeout: this.timeout
      });

      return response.data;
    } catch (error) {
      console.error('Music generation error:', error.message);
      throw this.handleError(error);
    }
  }

  // Extract audio features
  async extractFeatures(audioPath) {
    try {
      const formData = new FormData();
      formData.append('audio', fs.createReadStream(audioPath));

      const response = await axios.post(`${this.baseURL}/extract-features`, formData, {
        headers: formData.getHeaders(),
        timeout: 30000
      });

      return response.data;
    } catch (error) {
      console.error('Feature extraction error:', error.message);
      throw this.handleError(error);
    }
  }

  // Get waveform data
  async getWaveform(audioPath, options = {}) {
    try {
      const formData = new FormData();
      formData.append('audio', fs.createReadStream(audioPath));
      formData.append('samples', options.samples || 1000);

      const response = await axios.post(`${this.baseURL}/waveform`, formData, {
        headers: formData.getHeaders(),
        timeout: 30000
      });

      return response.data;
    } catch (error) {
      console.error('Waveform generation error:', error.message);
      throw this.handleError(error);
    }
  }

  // Get spectrogram data
  async getSpectrogram(audioPath, options = {}) {
    try {
      const formData = new FormData();
      formData.append('audio', fs.createReadStream(audioPath));
      
      const response = await axios.post(`${this.baseURL}/spectrogram`, formData, {
        headers: formData.getHeaders(),
        timeout: 30000,
        responseType: 'arraybuffer'
      });

      return response.data;
    } catch (error) {
      console.error('Spectrogram generation error:', error.message);
      throw this.handleError(error);
    }
  }

  // Apply audio effects
  async applyEffects(audioPath, effects) {
    try {
      const formData = new FormData();
      formData.append('audio', fs.createReadStream(audioPath));
      formData.append('effects', JSON.stringify(effects));

      const response = await axios.post(`${this.baseURL}/apply-effects`, formData, {
        headers: formData.getHeaders(),
        timeout: this.timeout
      });

      return response.data;
    } catch (error) {
      console.error('Effects application error:', error.message);
      throw this.handleError(error);
    }
  }

  // Error handler
  handleError(error) {
    if (error.response) {
      return new Error(error.response.data.message || 'Python AI service error');
    } else if (error.request) {
      return new Error('Python AI service not responding');
    } else {
      return new Error(error.message);
    }
  }
}

module.exports = new PythonAIService();
