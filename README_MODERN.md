# Modern OpenWebUI Tools and Functions

## Overview

This repository contains modern, secure, and performant tools and functions for OpenWebUI, designed to work with the latest versions of Ollama (v0.11.8+) and OpenWebUI (v0.6.26+).

## 🚀 Key Features

### Modern Architecture
- **Circuit Breaker Pattern**: Prevents cascade failures
- **Connection Pooling**: Optimized HTTP connections
- **Rate Limiting**: Prevents API abuse
- **Async-First Design**: High-performance concurrent operations

### Enhanced Security
- **SSL Verification**: Secure by default
- **Input Validation**: Pydantic-based validation
- **Error Handling**: Secure error messages
- **Access Control**: Granular permissions support

### Advanced Capabilities
- **Function Calling**: Full Ollama function calling support
- **Streaming Responses**: Real-time streaming capabilities
- **Multi-modal Support**: Image and text processing
- **Emotional Intelligence**: Context-aware responses
- **Memory Management**: Intelligent memory retention
- **Advanced Reasoning**: Quality assessment and improvement

### OpenWebUI Integration
- **v0.6.26+ Compatible**: Latest features support
- **Tool Server Integration**: Seamless tool integration
- **Custom Reasoning Tags**: Configurable reasoning detection
- **Granular Permissions**: Fine-grained access control

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Ollama**: v0.11.8 or higher
- **OpenWebUI**: v0.6.26 or higher
- **Memory**: 2GB RAM minimum (4GB recommended)
- **Storage**: 1GB free space

### Dependencies
All dependencies are listed in `requirements.txt` and will be installed automatically.

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd openwebui-tools
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Copy and edit configuration
cp config.yaml.example config.yaml
# Edit config.yaml with your settings
```

### 5. Set Environment Variables (Optional)
```bash
export OLLAMA_HOST=localhost
export OLLAMA_PORT=11434
export OLLAMA_MODEL=llama3.2:8b
```

## 🔧 Configuration

### Basic Configuration
The `config.yaml` file contains all configuration options. Key sections:

```yaml
# Ollama settings
ollama:
  host: "localhost"
  port: 11434
  default_model: "llama3.2:8b"

# Security settings
security:
  enable_ssl: true
  timeout: 30
  max_retries: 3

# Feature flags
features:
  enable_function_calling: true
  enable_streaming: true
  enable_emotional_intelligence: true
```

### Environment-Specific Configuration
```yaml
environments:
  development:
    logging:
      level: "DEBUG"
  production:
    logging:
      level: "WARNING"
    security:
      timeout: 60
```

## 📚 Usage

### Basic Usage

#### 1. Import and Initialize
```python
from modern_openwebui_base import ModernBaseFilter
from modern_find_learn_remember import ModernFindLearnRemember
from modern_luna_enhanced_capabilities import ModernLunaEnhancedCapabilities

# Initialize with default configuration
filter = ModernFindLearnRemember()

# Or with custom configuration
from modern_openwebui_base import ModernValves
valves = ModernValves(
    ollama_host="localhost",
    default_model="llama3.2:8b",
    enable_ssl=True
)
filter = ModernFindLearnRemember(valves)
```

#### 2. Process Messages
```python
import asyncio

async def main():
    # Message body from OpenWebUI
    body = {
        "messages": [
            {"role": "user", "content": "What's the latest news about AI?"}
        ]
    }
    
    # Process the message
    result = await filter.process_message(body)
    print(result["messages"][-1]["content"])

# Run the async function
asyncio.run(main())
```

#### 3. Using as Context Manager
```python
async def process_with_context():
    async with ModernFindLearnRemember() as filter:
        body = {"messages": [{"role": "user", "content": "Hello!"}]}
        result = await filter.process_message(body)
        return result
```

### Advanced Usage

#### Function Calling
```python
# Define tools for function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

# Use in chat completion
response = await filter.ollama_client.chat_completion(
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    stream=False
)
```

#### Streaming Responses
```python
# Enable streaming
response = await filter.ollama_client.chat_completion(
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

# Process streaming response
for chunk in response.get("stream_data", []):
    if chunk.get("done", False):
        break
    content = chunk.get("message", {}).get("content", "")
    if content:
        print(content, end="", flush=True)
```

#### Emotional Intelligence
```python
# Analyze emotional state
emotional_state = filter.emotional_system.analyze_emotional_state(
    "I'm really frustrated with this problem"
)

print(f"Detected emotion: {emotional_state.primary_emotion}")
print(f"Intensity: {emotional_state.intensity}")

# Generate empathetic response
empathetic_response = filter.emotional_system.generate_empathetic_response(
    emotional_state, "Let me help you solve this issue"
)
```

#### Memory Management
```python
# Add information to memory
filter.memory_system.add_memory(
    content="User asked about Python async programming",
    importance=0.8,
    context={"topic": "programming", "language": "python"}
)

# Retrieve relevant memories
memories = filter.memory_system.find_relevant_memories("async programming")
for memory in memories:
    print(f"Memory: {memory['content']}")
```

## 🔄 Migration from Legacy Tools

### Automatic Migration
The modern tools include automatic migration capabilities:

```python
# Legacy filter
from Find_Learn_Remember import Filter as LegacyFilter

# Modern filter with legacy compatibility
from modern_find_learn_remember import ModernFindLearnRemember

# Initialize with migration mode
filter = ModernFindLearnRemember()
filter.enable_legacy_compatibility = True

# Process legacy format messages
result = await filter.process_message(legacy_body)
```

### Manual Migration Steps

#### 1. Update Imports
```python
# Old
from Find_Learn_Remember import Filter

# New
from modern_find_learn_remember import ModernFindLearnRemember
```

#### 2. Update Configuration
```python
# Old
filter = Filter()
filter.valves.model = "luna-tic:base"

# New
from modern_openwebui_base import ModernValves
valves = ModernValves(default_model="llama3.2:8b")
filter = ModernFindLearnRemember(valves)
```

#### 3. Update API Calls
```python
# Old
result = filter.outlet(body, event_emitter, user)

# New
result = await filter.process_message(body, event_emitter, user)
```

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest test_modern_base.py

# Run async tests
pytest test_async_features.py
```

### Test Configuration
```bash
# Set test environment
export TESTING=true
export OLLAMA_HOST=localhost
export OLLAMA_PORT=11434

# Run tests with specific configuration
pytest --config=test_config.yaml
```

## 📊 Monitoring and Metrics

### Health Checks
```python
# Check system health
health_status = await filter.check_health()
print(f"System health: {health_status['status']}")
print(f"Ollama connection: {health_status['ollama']['status']}")
print(f"Memory usage: {health_status['memory']['usage_percent']}%")
```

### Metrics Collection
```python
# Get performance metrics
metrics = await filter.get_metrics()
print(f"Response time: {metrics['avg_response_time']}ms")
print(f"Success rate: {metrics['success_rate']}%")
print(f"Active connections: {metrics['active_connections']}")
```

### Prometheus Integration
```python
# Enable Prometheus metrics
from prometheus_client import start_http_server

# Start metrics server
start_http_server(9090)

# Metrics will be available at http://localhost:9090/metrics
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Ollama Connection Failed
```bash
# Check Ollama status
ollama list

# Check network connectivity
curl http://localhost:11434/api/tags

# Verify configuration
python -c "from modern_openwebui_base import ModernValves; print(ModernValves().ollama_base_url)"
```

#### 2. SSL Certificate Issues
```yaml
# In config.yaml
security:
  enable_ssl: false  # Disable SSL for development
  ssl_context: null
```

#### 3. Memory Issues
```yaml
# Reduce memory usage
performance:
  connection_pool_size: 20
  max_concurrent_requests: 5
  memory:
    max_short_term: 10
    max_long_term: 100
```

#### 4. Performance Issues
```yaml
# Optimize performance
performance:
  enable_caching: true
  cache_ttl: 600
  connection_pool_size: 200
  max_concurrent_requests: 50
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode in configuration
logging:
  level: "DEBUG"
  structured:
    enable: true
    include_context: true
```

## 🔒 Security Considerations

### SSL Configuration
```yaml
security:
  enable_ssl: true
  ssl_context:
    verify_mode: "CERT_REQUIRED"
    check_hostname: true
```

### Rate Limiting
```yaml
security:
  rate_limit:
    max_calls: 100
    time_window: 60
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 60
```

### Input Validation
```python
# All inputs are automatically validated using Pydantic
from pydantic import ValidationError

try:
    filter = ModernFindLearnRemember(config)
except ValidationError as e:
    print(f"Configuration error: {e}")
```

## 📈 Performance Optimization

### Connection Pooling
```yaml
performance:
  connection_pool_size: 100
  max_concurrent_requests: 10
  enable_cleanup_closed: true
```

### Caching Strategy
```yaml
performance:
  enable_caching: true
  cache_ttl: 300
  cache_strategy: "lru"  # or "ttl", "adaptive"
```

### Memory Management
```yaml
performance:
  memory:
    cleanup_interval: 3600
    max_short_term: 20
    max_long_term: 1000
    consolidation_enabled: true
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black .
isort .

# Run linting
flake8 .
mypy .
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

### Testing Guidelines
- Maintain 90%+ test coverage
- Test both success and failure scenarios
- Include integration tests for complex features
- Test async functionality thoroughly

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama Team**: For the excellent local LLM framework
- **OpenWebUI Team**: For the powerful web interface
- **Open Source Community**: For the tools and libraries that made this possible

## 📞 Support

### Getting Help
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Documentation**: Check this README and inline code documentation

### Community
- **Discord**: Join our Discord server for real-time support
- **Matrix**: Join our Matrix room for community discussions
- **Email**: Contact us at support@example.com

---

**Note**: This is a modern rewrite of the original OpenWebUI tools. For legacy support, please refer to the original files or use the migration compatibility mode.