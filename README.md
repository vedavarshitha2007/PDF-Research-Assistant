# 📄 PDF Research Assistant using RAG

An AI-powered application that allows users to upload PDF documents and ask questions about their content. It retrieves relevant information from the document and uses Google Gemini to generate contextual answers with source page references.

## 🚀 Features

- Upload and process PDF documents.
- Extract text using PyPDFLoader.
- Split documents into manageable text chunks.
- Retrieve relevant chunks using TF-IDF and cosine similarity.
- Generate contextual answers using Google Gemini.
- Display source page references.
- Maintain question history during the session.

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- PyPDFLoader
- Scikit-learn (TF-IDF and cosine similarity)
- Google Gemini API

## ⚙️ How It Works

1. User uploads a PDF document.
2. Text is extracted from the PDF.
3. Extracted text is divided into smaller chunks.
4. TF-IDF and cosine similarity identify relevant chunks for the question.
5. Retrieved context is passed to Google Gemini.
6. Gemini generates an answer based on the retrieved information.
7. The application displays the answer and source pages.

## 🏗️ Architecture

PDF Upload → Text Extraction → Text Chunking → TF-IDF Vectorization → Similarity-Based Retrieval → Gemini Answer Generation → Answer with Sources

## 💻 Run Locally

Clone the repository:

```bash
git clone https://github.com/vedavarshitha2007/PDF-Research-Assistant.git
```

Navigate to the project folder:

```bash
cd PDF-Research-Assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure your Gemini API key using an environment variable or Streamlit secrets. Never commit API keys to GitHub.

Run the application:

```bash
streamlit run app.py
```

## 🔮 Future Enhancements

- Support for multiple PDF documents.
- Improved semantic retrieval using embedding models.
- Conversation memory across sessions.
- More advanced source citation and retrieval.

