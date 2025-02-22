# Job Application Form Generator

A full-stack web application for generating and submitting job application forms with local file storage.

## Prerequisites

- Python 3.12+
- Node.js 18+
- Poetry (Python package manager)
- npm (Node.js package manager)

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Start the backend server:
```bash
poetry run fastapi dev app/main.py
```

The backend will run on http://localhost:8000

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file with the following content:
```
VITE_API_URL=http://localhost:8000
```

4. Start the development server:
```bash
npm run dev
```

The frontend will run on http://localhost:5173

## Features

- Dynamic form generation from JSON configuration
- Support for various field types (text, email, tel, select, date, file, textarea)
- Form validation (required fields, email format)
- Local file storage for form submissions
- Responsive UI with shadcn/ui components

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI application
│   │   ├── models.py        # Pydantic models
│   │   ├── utils.py         # Utility functions
│   │   └── config.py        # Configuration settings
│   └── data/
│       └── submissions/     # Form submissions storage
└── frontend/
    ├── src/
    │   ├── components/      # React components
    │   ├── lib/             # Utilities and API client
    │   └── App.tsx         # Main application component
    └── .env                # Environment configuration
```

## Form Submission Storage

Form submissions are stored as JSON files in the `backend/data/submissions` directory. Each submission is saved with a timestamp-based filename for uniqueness.
