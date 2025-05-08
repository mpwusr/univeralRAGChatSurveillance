# UniveralRAGChatSurveillance

A Retrieval-Augmented Generation (RAG) solution for automating retail security incident monitoring using generative AI, image segmentation, and vector retrieval. This project integrates real-time surveillance data with natural language processing to assist security personnel in identifying suspicious activities, assessing risks, and responding to incidents efficiently.

## Overview

The UniveralRAGChatSurveillance system leverages advanced AI models and computer vision to enhance physical security monitoring in retail environments. By combining image segmentation (YOLO, Mask R-CNN), multimodal embeddings (CLIP/BLIP), and RAG-based reasoning (Grok-2, GPT-4o, Cohere, LLaMA), it automates manual monitoring tasks, reduces response times, and enables proactive security measures.

### Key Features
- **Real-Time Entity Detection**: Uses YOLO and Mask R-CNN for identifying entities (e.g., people, vehicles).
- **Multimodal Data Processing**: Embeds images and text using CLIP/BLIP.
- **Vector Retrieval**: Stores embeddings in Pinecone for fast similarity search.
- **Natural Language Interface**: Supports queries like "Is there suspicious activity at the gate?" via CLI or Tkinter GUI.
- **Multi-LLM Integration**: Switches between Grok-2, GPT-4o, Command-R, LLaMA, and Ollama.
- **Scalability**: Supports Kafka-based live feeds and web dashboards.

## Repository Structure
- **`create_daily_report.py`**: Generates daily security reports.
- **`preprocess_combined_models.py`**: Preprocesses images using YOLO and Mask R-CNN.
- **`preprocess_friendly_models.py`**: Identifies friendly entities.
- **`ragchat.py`**: Core RAG pipeline, CLI/GUI interaction, and LLM handling.

## Installation

### Prerequisites
- Python 3.10+
- API keys for Pinecone, XAI (Grok), OpenAI, Cohere, Hugging Face
- Dependencies: `torch`, `transformers`, `pinecone-client`, `requests`, `python-dotenv`, `nltk`, `langchain-huggingface`, `huggingface_hub`

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mpwusr/univeralRAGChatSurveillance.git
   cd univeralRAGChatSurveillance
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file:
   ```bash
   PINECONE_API_KEY=your_pinecone_api_key
   XAI_API_KEY=your_xai_api_key
   OPENAI_API_KEY=your_openai_api_key
   CO_API_KEY=your_cohere_api_key
   HF_API_KEY=your_huggingface_api_key
   ```

4. **Initialize Pinecone Index**:
   ```bash
   python setup_pinecone.py
   ```

## Hugging Face LLaMA Integration
The project integrates Meta’s LLaMA models (e.g., `meta-llama/Llama-2-7b-chat-hf`) via Hugging Face’s Inference API for text generation and chat capabilities.

### Setup
1. **Install Dependencies**:
   ```bash
   pip install langchain-huggingface huggingface_hub
   ```
2. **Log in to Hugging Face**:
   ```bash
   huggingface-cli login
   ```
3. **Request Access**:
   - Visit `https://huggingface.co/meta-llama/Llama-2-7b-chat-hf` and request access.
4. **Usage in `ragchat.py`**:
   - LLaMA is configured in `ragchat.py`:
     ```python
     self.hf_endpoint = HuggingFaceEndpoint(
         repo_id="meta-llama/Llama-2-7b-chat-hf",
         max_new_tokens=1024,
         temperature=0.7,
         top_p=0.9,
         huggingfacehub_api_token=self.hf_api_key
     )
     self.hf_client = ChatHuggingFace(llm=self.hf_endpoint)
     ```
   - Run with LLaMA:
     ```bash
     python ragchat.py --service huggingface --model meta-llama/Llama-2-7b-chat-hf
     ```

## Usage

1. **Populate Pinecone**:
   ```bash
   python populate_pinecone.py
   ```

2. **Preprocess Images**:
   - Combined models:
     ```bash
     python preprocess_combined_models.py
     ```
   - Friendly entities:
     ```bash
     python preprocess_friendly_models.py
     ```

3. **Run the Application**:
   - CLI Mode:
     ```bash
     python ragchat.py --service grok --model grok-2
     ```
     or for LLaMA:
     ```bash
     python ragchat.py --service huggingface --model meta-llama/Llama-2-7b-chat-hf
     ```

4. **Generate Daily Reports**:
   ```bash
   python create_daily_report.py
   ```

5. **Example Queries**:
   - "Is there suspicious activity at the gate?"
   - "Did anyone enter the restricted area last night?"
   - "What are recent trends in perimeter breaches?"

### Switching Services and Models
- **Command-Line**:
  ```bash
  python ragchat.py --service <service> --model <model>
  ```
  Examples:
  - `python ragchat.py --service grok --model grok-2`
  - `python ragchat.py --service huggingface --model meta-llama/Llama-2-7b-chat-hf`
  - Supported services: `grok`, `openai`, `cohere`, `huggingface`, `ollama`
- **Interactive**:
  - Type `switch to <service>` (e.g., `switch to huggingface`).
  - Type `set model <model>` (e.g., `set model meta-llama/Llama-2-7b-chat-hf`).
  - Example:
    ```
    switch to huggingface
    set model meta-llama/Llama-2-7b-chat-hf
    ```

### Quitting or Exiting
- **CLI Mode**:
  - Type `exit` or `quit`:
    ```
    [Grok:grok-2] How can I assist you today? exit
    Goodbye!
    ```
## Future Enhancements
- Real-time Kafka feed processing.
- Web-based dashboard with live alerts.
- RFID/smart lock integration.
- LLaMA fine-tuning for security contexts.

## Contributing
Submit issues or pull requests, adhering to PEP 8 and including unit tests.

## License
Apache-2.0 License.

## Contact
Michael Perry Williams at mw00066@vt.edu or [Saddle River Consulting](https://saddleriverconsulting.com).