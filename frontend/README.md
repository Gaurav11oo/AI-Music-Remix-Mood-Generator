# Frontend Application - SonicAI Music Remix

Complete Next.js 14 frontend application with interactive UI components.

## Features Implemented

### Pages
- Landing Page with animated hero section
- Login Page with authentication
- Register Page with validation
- Dashboard Page with full functionality

### Components

#### Audio Components
- **AudioUploader**: Drag-and-drop file upload with validation
- **AudioList**: Grid view of uploaded audio files with actions
- **StemSeparation**: Interactive stem separation interface
- **MoodClassifier**: Mood analysis with visual representation
- **MusicGenerator**: Text-to-music generation interface

#### UI Components (Shadcn-based)
- Button with variants
- Card with sections
- Input fields
- Progress bars
- Slider controls
- Tabs navigation
- Dropdown menus
- Labels

### State Management
- Zustand stores for auth, audio, jobs, and UI state
- Persistent authentication state
- Real-time job status updates

### Features
- JWT authentication flow
- File upload with drag-and-drop
- Real-time progress tracking
- Toast notifications
- Responsive design
- Dark mode by default
- Smooth animations with Framer Motion

## Installation

```bash
npm install
```

## Environment Variables

Create `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:5000/api
```

## Development

```bash
npm run dev
```

Open http://localhost:3000

## Build

```bash
npm run build
npm start
```

## Dependencies

### Core
- Next.js 14
- React 18
- TypeScript

### UI & Styling
- Tailwind CSS
- Framer Motion
- Shadcn UI (Radix UI)
- Lucide React Icons

### State & API
- Zustand
- Axios
- React Hot Toast

### Audio
- WaveSurfer.js
- Tone.js

### Forms
- React Hook Form
- Zod validation
- React Dropzone

### Charts
- Chart.js
- Recharts

## Project Structure

```
src/
├── app/
│   ├── dashboard/
│   │   └── page.tsx
│   ├── login/
│   │   └── page.tsx
│   ├── register/
│   │   └── page.tsx
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   ├── audio/
│   │   ├── AudioList.tsx
│   │   ├── AudioUploader.tsx
│   │   ├── MoodClassifier.tsx
│   │   ├── MusicGenerator.tsx
│   │   └── StemSeparation.tsx
│   └── ui/
│       ├── button.tsx
│       ├── card.tsx
│       ├── dropdown-menu.tsx
│       ├── input.tsx
│       ├── label.tsx
│       ├── progress.tsx
│       ├── slider.tsx
│       └── tabs.tsx
└── lib/
    ├── api.ts
    ├── store.ts
    └── utils.ts
```

## Usage Guide

### Authentication
1. Register a new account at /register
2. Login at /login
3. Access dashboard at /dashboard

### Upload Audio
1. Go to Upload tab
2. Drag and drop audio file or click to browse
3. Supported formats: MP3, WAV, FLAC, OGG, M4A, AAC
4. Max size: 100MB

### Stem Separation
1. Select an audio file from library
2. Go to Stems tab
3. Click "Separate Stems"
4. Wait for processing
5. Download separated stems

### Mood Classification
1. Select an audio file
2. Go to Mood tab
3. Click "Classify Mood"
4. View mood distribution and audio features

### Music Generation
1. Go to Generate tab
2. Enter text description
3. Adjust duration and creativity
4. Click "Generate Music"
5. Download generated audio

## Customization

### Theme
Edit `src/app/globals.css` to customize colors:

```css
:root {
  --primary: 262 83% 58%;
  --secondary: 210 40% 96.1%;
  /* ... */
}
```

### Components
All UI components are in `src/components/ui` and can be customized.

## API Integration

The app integrates with the backend API using Axios client in `src/lib/api.ts`.

All API calls include:
- Automatic JWT token handling
- Error handling
- Response interceptors
- Loading states

## Performance

- Code splitting with Next.js
- Lazy loading of components
- Optimized images
- Memoized expensive operations
- Debounced API calls

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Deployment

### Vercel (Recommended)
1. Push to GitHub
2. Import to Vercel
3. Set environment variables
4. Deploy

### Other Platforms
```bash
npm run build
```

Deploy the `.next` folder.

## Troubleshooting

### Authentication Issues
- Clear localStorage
- Check API URL in .env.local
- Verify backend is running

### Upload Failures
- Check file size (max 100MB)
- Verify file format
- Check network connection

### Build Errors
- Delete node_modules and .next
- Run npm install
- Check TypeScript errors

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## License

MIT License
