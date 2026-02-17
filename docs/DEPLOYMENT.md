# Deployment Guide

## Deployment Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Vercel (Frontend)                  │
│              Next.js 14 Application                 │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │  Pages   │  │ API Proxy│  │  Assets  │         │
│  └──────────┘  └──────────┘  └──────────┘         │
└────────────────────┬────────────────────────────────┘
                     │ HTTPS
                     ▼
┌─────────────────────────────────────────────────────┐
│            AWS EC2 / GCP VM (Backend)               │
│                                                     │
│  ┌────────────────────────────────────────┐        │
│  │        Express.js API Server           │        │
│  │         (Node.js 18+)                  │        │
│  │  Port: 5000                            │        │
│  └───────────┬────────────────────────────┘        │
│              │                                      │
│  ┌───────────┴────────────────────────────┐        │
│  │    Bull Queues (Background Jobs)       │        │
│  │    ┌────────┐  ┌────────┐  ┌────────┐ │        │
│  │    │ Stems  │  │  Mood  │  │ Remix  │ │        │
│  │    └────────┘  └────────┘  └────────┘ │        │
│  └───────────┬────────────────────────────┘        │
│              │                                      │
│  ┌───────────┴────────────────────────────┐        │
│  │      Python AI Service (Flask)         │        │
│  │         Port: 5001                     │        │
│  │  ┌────────────┐  ┌────────────┐       │        │
│  │  │   Demucs   │  │  Librosa   │       │        │
│  │  └────────────┘  └────────────┘       │        │
│  └────────────────────────────────────────┘        │
│                                                     │
│  ┌────────────────────────────────────────┐        │
│  │         Redis (Queue Storage)          │        │
│  │            Port: 6379                  │        │
│  └────────────────────────────────────────┘        │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│     Managed PostgreSQL (AWS RDS / GCP SQL)          │
│              Port: 5432                             │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐   │
│  │ Users  │  │ Audio  │  │  Jobs  │  │ Moods  │   │
│  └────────┘  └────────┘  └────────┘  └────────┘   │
└─────────────────────────────────────────────────────┘

                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│         File Storage (Local/NFS/S3-Compatible)      │
│            /uploads, /temp, /outputs                │
└─────────────────────────────────────────────────────┘
```

---

## Frontend Deployment (Vercel)

### Prerequisites
- Vercel account
- GitHub repository
- Environment variables configured

### Steps

1. **Push code to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/music-remix-app.git
git push -u origin main
```

2. **Connect to Vercel**
- Visit [vercel.com](https://vercel.com)
- Import your GitHub repository
- Select the `frontend` directory as root

3. **Configure Environment Variables**
```
NEXT_PUBLIC_API_URL=https://api.yourdomain.com/api
NEXT_PUBLIC_WS_URL=wss://api.yourdomain.com
```

4. **Deploy**
- Vercel will automatically build and deploy
- Custom domain can be configured in settings

### Build Configuration
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "installCommand": "npm install",
  "framework": "nextjs"
}
```

---

## Backend Deployment (AWS EC2 / GCP Compute Engine)

### Option 1: AWS EC2

#### 1. Launch EC2 Instance
```bash
# Instance type: t3.medium or larger (2 vCPU, 4GB RAM minimum)
# OS: Ubuntu 22.04 LTS
# Storage: 30GB+ SSD
# Security Group: Allow ports 22, 80, 443, 5000
```

#### 2. Connect and Setup
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Python 3.9+
sudo apt install -y python3.9 python3-pip python3-venv

# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Install Redis
sudo apt install -y redis-server

# Install FFmpeg
sudo apt install -y ffmpeg

# Install PM2 for process management
sudo npm install -g pm2
```

#### 3. Deploy Application
```bash
# Clone repository
git clone https://github.com/yourusername/music-remix-app.git
cd music-remix-app

# Setup Backend
cd backend
npm install
cp .env.example .env
# Edit .env with production values
nano .env

# Run migrations
npm run migrate

# Setup Python AI Service
cd python-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate

cd ..

# Start services with PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

#### 4. Configure NGINX
```bash
sudo apt install -y nginx

# Create NGINX config
sudo nano /etc/nginx/sites-available/music-remix
```

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    client_max_body_size 100M;
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/music-remix /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 5. Setup SSL with Let's Encrypt
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourdomain.com
```

### Option 2: GCP Compute Engine

Similar steps, using:
```bash
# Create instance
gcloud compute instances create music-remix-api \
  --machine-type=e2-medium \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB \
  --tags=http-server,https-server
```

---

## Database Deployment

### AWS RDS (PostgreSQL)

1. **Create RDS Instance**
- Engine: PostgreSQL 14+
- Instance class: db.t3.micro (free tier) or larger
- Storage: 20GB SSD
- Enable automated backups
- VPC: Same as EC2 instance

2. **Configure Security Group**
- Allow PostgreSQL (5432) from EC2 security group

3. **Connection String**
```
postgresql://username:password@your-rds-endpoint:5432/music_remix_db
```

### GCP Cloud SQL

```bash
gcloud sql instances create music-remix-db \
  --database-version=POSTGRES_14 \
  --tier=db-f1-micro \
  --region=us-central1
```

---

## Redis Deployment

### AWS ElastiCache
- Engine: Redis 6.x
- Node type: cache.t3.micro
- VPC: Same as EC2

### GCP Memorystore
```bash
gcloud redis instances create music-remix-redis \
  --size=1 \
  --region=us-central1
```

### Or use Redis Cloud (Managed)
- Sign up at [redis.com](https://redis.com)
- Create free database (30MB)
- Get connection string

---

## Environment Variables (Production)

### Backend (.env)
```env
NODE_ENV=production
PORT=5000
API_BASE_URL=https://api.yourdomain.com

# Database (AWS RDS)
DB_HOST=your-rds-endpoint.amazonaws.com
DB_PORT=5432
DB_NAME=music_remix_db
DB_USER=admin
DB_PASSWORD=your-secure-password

# Redis (ElastiCache or Redis Cloud)
REDIS_HOST=your-redis-endpoint.amazonaws.com
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# JWT
JWT_SECRET=your-production-secret-key-min-32-chars
JWT_EXPIRY=7d

# File Storage
UPLOAD_DIR=/var/www/music-remix/uploads
TEMP_DIR=/var/www/music-remix/temp

# Python AI Service
PYTHON_AI_URL=http://localhost:5001

# CORS
CORS_ORIGIN=https://yourdomain.com
```

---

## PM2 Ecosystem Config

Create `backend/ecosystem.config.js`:

```javascript
module.exports = {
  apps: [
    {
      name: 'music-remix-api',
      script: './src/app.js',
      instances: 2,
      exec_mode: 'cluster',
      env: {
        NODE_ENV: 'production',
      },
      error_file: './logs/err.log',
      out_file: './logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    },
    {
      name: 'audio-worker',
      script: './src/workers/audioWorker.js',
      instances: 1,
      env: {
        NODE_ENV: 'production',
      },
    },
    {
      name: 'python-ai-service',
      script: './python-ai/venv/bin/python',
      args: './python-ai/app.py',
      instances: 1,
      env: {
        FLASK_ENV: 'production',
      },
    },
  ],
};
```

---

## Monitoring & Logging

### PM2 Monitoring
```bash
pm2 monit
pm2 logs
pm2 status
```

### Application Logging
- Winston for structured logging
- CloudWatch Logs (AWS) or Cloud Logging (GCP)

### Database Monitoring
- AWS RDS Performance Insights
- GCP Cloud SQL Insights

### Alerts
- Setup CloudWatch Alarms for:
  - CPU usage > 80%
  - Memory usage > 85%
  - Disk usage > 90%
  - API errors > 100/hour

---

## Backup Strategy

### Database Backups
- AWS RDS: Automated daily backups (7-day retention)
- Manual snapshots before major updates

### File Backups
```bash
# Daily cron job
0 2 * * * tar -czf /backups/uploads-$(date +\%Y\%m\%d).tar.gz /var/www/music-remix/uploads
```

---

## Scaling Considerations

### Horizontal Scaling
- Multiple EC2 instances behind load balancer
- Shared file storage (NFS or S3)
- Redis cluster for queue distribution

### Vertical Scaling
- Upgrade instance types as needed
- Add GPU instances for AI processing

### CDN
- CloudFront (AWS) or Cloud CDN (GCP) for static assets
- Cache audio previews and spectrograms

---

## Security Checklist

- [ ] Enable HTTPS only (SSL/TLS)
- [ ] Configure firewall rules (Security Groups)
- [ ] Enable database encryption at rest
- [ ] Use IAM roles for AWS resource access
- [ ] Rotate JWT secrets regularly
- [ ] Implement rate limiting
- [ ] Enable CORS with specific origins
- [ ] Regular security updates
- [ ] Disable unnecessary ports
- [ ] Use environment variables for secrets

---

## Troubleshooting

### Check Service Status
```bash
pm2 status
sudo systemctl status nginx
sudo systemctl status postgresql
sudo systemctl status redis
```

### View Logs
```bash
pm2 logs music-remix-api
pm2 logs python-ai-service
sudo tail -f /var/log/nginx/error.log
```

### Restart Services
```bash
pm2 restart all
sudo systemctl restart nginx
```

### Database Connection Issues
```bash
# Test PostgreSQL connection
psql -h your-db-host -U admin -d music_remix_db

# Check Redis connection
redis-cli -h your-redis-host ping
```

---

## Cost Estimation (Monthly)

### Minimal Setup
- Vercel (Frontend): Free tier
- AWS EC2 t3.small: ~$15
- AWS RDS db.t3.micro: ~$15
- AWS ElastiCache t3.micro: ~$12
- **Total: ~$42/month**

### Production Setup
- Vercel Pro: $20
- AWS EC2 t3.medium: ~$30
- AWS RDS db.t3.small: ~$25
- AWS ElastiCache t3.small: ~$25
- **Total: ~$100/month**

### High-Traffic Setup
- Vercel Pro: $20
- AWS EC2 c5.large (2x): ~$130
- AWS RDS db.m5.large: ~$150
- AWS ElastiCache m5.large: ~$120
- Load Balancer: ~$20
- **Total: ~$440/month**
