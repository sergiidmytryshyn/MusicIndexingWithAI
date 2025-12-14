# MIWA - Music Indexing with AI

**MIWA** (Music Indexing with AI) is an intelligent music search system that helps users find songs using natural language queries. Instead of requiring exact song titles or artist names, users can describe songs using partial lyrics, mood, genre, artist characteristics, location, or any combination of these attributes.

## 🎯 Features

- **Natural Language Query Processing**: Extract structured information from conversational queries using Claude AI
- **Multi-Modal Search**: Search by:
  - Song title keywords
  - Lyrics (exact keywords or semantic similarity)
  - Artist name and characteristics
  - Genre
  - Release year (ranges)
  - Geographic location (country or region)
  - Album information
  - Featured artists
- **GraphRAG Integration**: Powered by Neo4j graph database for efficient relationship-based retrieval
- **Hybrid Retrieval**: Combines full-text search (Lucene) and vector embeddings (Jina embeddings) for optimal results
- **Interactive Streamlit Demo**: User-friendly web interface for testing queries
- **Benchmarking & Evaluation**: Tools for testing extraction accuracy and retrieval performance

## 🏗️ Architecture

### System Components

1. **LLM Extraction Layer** (`app.py`)
   - Uses Anthropic Claude API with COSTAR-formatted prompts
   - Extracts structured JSON from natural language queries
   - Outputs XML for reliable parsing

2. **GraphRAG Database** (Neo4j)
   - Stores tracks, artists, albums, genres, locations, lyrics, and descriptions
   - Full-text indexes for keyword search
   - Vector indexes for semantic similarity search
   - Relationship-based queries for complex filtering

3. **Retrieval Engine** (`app.py`, `calculate_accuracy.py`)
   - Multi-stage filtering and scoring system
   - Combines multiple score types (title, lyrics, artist, album)
   - Softmax normalization for balanced scoring
   - Returns top-K results ranked by relevance

4. **Evaluation Tools**
   - `benchmark_extraction.py`: Tests field extraction accuracy
   - `calculate_accuracy.py`: Evaluates retrieval performance (Top-1, Top-3, Top-10, Top-15)
   - `generate_prompts.py`: Generates diverse test prompts using OLLAMA

## 📋 Prerequisites

- Python 3.8+
- Neo4j database (version 5.x or later)
- Anthropic API key (for Claude)
- (Optional) OLLAMA (for prompt generation)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MusicIndexingWithAI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # Anthropic API (required)
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   
   # Neo4j Connection (required)
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password_here
   ```

4. **Set up Neo4j Database**
   
   - Install and start Neo4j
   - Create the graph structure using `src/create_graph.py`
   - Load your music data into Neo4j
   - The system will automatically create necessary indexes

## 💻 Usage

### Running the Streamlit Demo

```bash
streamlit run app.py
```

The demo will open in your browser. You can:
- Enter natural language queries about songs
- View extracted structured information
- See retrieval results and AI-generated suggestions
- Enable debug mode to inspect extraction details

### Example Queries

- "Find songs with 'love' in the title from the 2000s"
- "Rock songs by Norwegian artists with lyrics about mountains"
- "Songs featuring Drake released after 2015"
- "Pop songs from Europe with lyrics mentioning 'dancing'"
- "Songs by artists founded in the 1960s from the United Kingdom"

### Benchmarking Extraction

Test the extraction system on generated prompts:

```bash
python3 benchmark_extraction.py
```

This will:
- Load prompts from `data_parsing/data/generated_prompts.json`
- Extract fields for each prompt
- Generate a benchmark report
- Update the prompts file with extracted JSON

### Calculating Retrieval Accuracy

Evaluate retrieval performance:

```bash
python3 calculate_accuracy.py
```

This will:
- Load prompts with expected track information
- Run retrieval for each prompt
- Calculate Top-1, Top-3, Top-10, Top-15 accuracy
- Generate accuracy results JSON files

### Generating Test Prompts

Generate diverse prompts for testing:

```bash
python3 generate_prompts.py
```

**Note**: Requires OLLAMA running locally with the `qwen3:14b` model.

## 📁 Project Structure

```
MusicIndexingWithAI/
├── app.py                      # Main Streamlit application
├── benchmark_extraction.py     # Extraction benchmarking script
├── calculate_accuracy.py       # Retrieval accuracy evaluation
├── generate_prompts.py         # Test prompt generation (OLLAMA)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/                       # Sample data files
│   ├── artists_locations_countries_only.json
│   ├── genres_sample.json
│   └── locations_sample.json
│
├── data_parsing/               # Data processing scripts
│   ├── data/
│   │   ├── generated_prompts.json    # Generated test prompts
│   │   ├── sample_100.json           # Sample tracks for prompt generation
│   │   └── datascripts/              # Data processing utilities
│   ├── eda/                    # Exploratory data analysis
│   └── utils/                  # Parsing utilities
│
├── results/                    # Evaluation results
│   ├── benchmark_report_*.json
│   └── accuracy_results_*.json
│
└── src/                        # Source code
    └── create_graph.py         # Neo4j graph creation script
```

## 🔧 Key Components

### Extraction System

The extraction system uses a COSTAR-formatted prompt to extract structured information:

```json
{
  "track": {
    "title_keywords": ["love", "heart"],
    "year_from": 2000,
    "year_to": 2010,
    "genres": ["rock", "pop"],
    "lyrics_keywords": ["tears"],
    "lyrics_text": "songs about crying and sadness"
  },
  "artist": {
    "name_keywords": ["beatles"],
    "country": "United Kingdom",
    "region": null,
    "description_text": "legendary rock band"
  },
  "features": ["Drake"],
  "album": {
    "name_keywords": ["thriller"]
  }
}
```

### Retrieval System

The retrieval system uses a multi-stage approach:

1. **Filtering**: Apply numeric filters (year ranges, views) first
2. **Scoring**: Calculate relevance scores for:
   - Title match (full-text search)
   - Lyrics match (full-text + vector similarity)
   - Artist name match (full-text search)
   - Artist description (vector similarity)
   - Album match (full-text search)
3. **Ranking**: Combine normalized scores and return top-K results

## 👥 Contributors
- [Serhii Dmytryshyn](https://github.com/sergiidmytryshyn) - GraphRAG and retrieval implementation, Neo4j integration
- [Zakhar Kohut](https://github.com/zahar-kohut-ucu) - Evaluation tools, benchmarking scripts, LLM integration and prompt generation
- [Andrii Kravchuk](https://github.com/movchun567) - Data parsing and preprocessing

## 🙏 Acknowledgments

- Anthropic for Claude API
- Neo4j for graph database
- Jina AI for embeddings model

