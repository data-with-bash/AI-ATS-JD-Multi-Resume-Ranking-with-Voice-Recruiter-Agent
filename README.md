# 🤖 AI-Powered ATS with Voice Recruiter Agent

An intelligent Applicant Tracking System that uses semantic search and RAG (Retrieval-Augmented Generation) to rank candidates and enable recruiters to interact with candidate data through voice or text queries.

## 🌟 Features

### 1. **Multi-Resume Processing**
- Upload one Job Description (JD) and multiple candidate resumes (PDF)
- Automatic text extraction and chunking for optimal vector storage
- Stores all data in Qdrant vector database for semantic search

### 2. **Semantic ATS Ranking**
- Ranks candidates by semantic fit score (0-100) against the JD
- Uses vector similarity between JD and resume embeddings
- Displays ranked shortlist with explainable scoring
- Inspect individual candidate resumes

### 3. **Voice Recruiter Agent** 🎙️
- Ask questions about candidates using voice input or text
- Powered by RAG: retrieves relevant resume excerpts from Qdrant
- Groq LLM generates accurate, context-aware answers
- Examples:
  - "Who are the top 3 candidates for backend Python?"
  - "Which candidates have cloud experience?"
  - "Summarize the best fit for this role"

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python web framework)
- **Vector Database**: Qdrant Cloud (semantic search)
- **LLM**: Groq (Llama 3.1 70B)
- **Embeddings**: HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **RAG Framework**: LangChain
- **Voice Input**: streamlit-mic-recorder
- **PDF Processing**: PyPDF2

## 📦 Installation

### Prerequisites
- Python 3.8+
- Qdrant Cloud account (free tier: https://cloud.qdrant.io)
- Groq API key (free: https://console.groq.com)

### Step 1: Clone the repository

### Step 2: Install dependencies

### Step 3: Configure environment variables
Create a `.env` file in the project root:

GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here


## 🚀 Usage

### Run the application
streamlit run app.py




The app will open in your browser at `http://localhost:8501`

### Workflow

1. **Upload JD and Resumes**
   - Upload a Job Description (PDF or paste text)
   - Upload multiple candidate resumes (PDF files)
   - Click "Process JD + Resumes"

2. **View ATS Ranking**
   - See ranked shortlist with fit scores
   - Inspect individual candidate resumes

3. **Use Voice Recruiter Agent**
   - Click microphone button to speak your question
   - OR type your question in the text box
   - Get AI-powered answers with source citations

## 📁 Project Structure

ai-ats-voice-recruiter/
├── app.py # Main Streamlit application
├── .env # Environment variables (create this)
├── requirements.txt # Python dependencies (optional)
└── README.md # This file


## 🎯 Use Cases

- **HR Teams**: Screen 100+ candidates in minutes instead of hours
- **Recruiters**: Ask natural language questions about candidate pools
- **Hiring Managers**: Quickly identify best-fit candidates for technical roles
- **Hackathons**: Demo-ready AI recruitment solution

## 🔐 Privacy & Security

- All API keys stored locally in `.env` (never committed to Git)
- Resume data stored in your private Qdrant Cloud cluster
- No data shared with third parties
- Resumes can be deleted anytime from Qdrant dashboard

## 🚧 Roadmap / Future Enhancements

- [ ] Add bias detection and fairness metrics
- [ ] Support for multiple JDs (compare candidates across roles)
- [ ] Export ranked candidates to CSV
- [ ] Candidate skill gap analysis
- [ ] Email integration for candidate outreach
- [ ] Support for Word documents and plain text resumes

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License

MIT License - feel free to use this project for personal or commercial purposes.

## 👨‍💻 Built By

**Your Name**
- GitHub: data_with_bash
- LinkedIn:https://www.linkedin.com/in/suman-sarkar-ai-engineer/
- Email: mr.sarkar4478987656@gmail.com

**Built for**: AI Memory Hackathon - November 2025

## 🙏 Acknowledgments

- Groq for blazing-fast LLM inference
- Qdrant for powerful vector search
- LangChain for RAG framework
- Streamlit for rapid UI development

---

⭐ Star this repo if you found it helpful!
