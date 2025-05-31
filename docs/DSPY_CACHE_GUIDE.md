# DSPy Cache Management Guide

This guide explains how to clear and manage DSPy LLM call cache in your fastworkflow project.

## 🔍 Understanding DSPy Caching

DSPy implements caching to avoid repeated identical LLM calls, which:
- **Speeds up development** - Cached responses return instantly
- **Saves API costs** - Repeated calls don't hit the API
- **Can interfere with testing** - You might want fresh responses each time

## 🗑️ Methods to Clear DSPy Cache

### **1. Programmatic Cache Clearing (Recommended)**

#### In Your Agent Code
```python
from fastworkflow.run_agent.agent_module import clear_dspy_cache, configure_dspy_cache

# Clear cache completely (disables all caching)
clear_dspy_cache()

# Or configure cache settings
configure_dspy_cache(enable_cache=False)  # Disable caching
configure_dspy_cache(enable_cache=True)   # Enable caching (default)
```

#### When Initializing Agent
```python
# Clear cache when initializing agent
react_agent = initialize_dspy_agent(
    workflow_session, 
    LLM_AGENT, 
    LITELLM_API_KEY_AGENT,
    clear_cache=True  # 🗑️ This clears cache before initialization
)
```

### **2. Using DSPy's Built-in Cache Configuration**

```python
import dspy

# Disable all caching
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
    enable_litellm_cache=False
)

# Re-enable caching later
dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    enable_litellm_cache=False
)
```

### **3. Command Line Cache Management**

Use the standalone cache utility:

```bash
# Clear all DSPy caches
python dspy_cache_utils.py clear

# Clear only disk cache files
python dspy_cache_utils.py clear-disk

# Reset cache to default settings
python dspy_cache_utils.py reset

# Check current cache status
python dspy_cache_utils.py status
```

### **4. Manual Cache Directory Deletion**

Find and delete cache directories manually:

```bash
# Common DSPy cache locations
rm -rf ~/.cache/dspy/
rm -rf ~/.dspy_cache/
rm -rf ./.dspy_cache/

# Or find all cache directories
find ~ -name "*dspy*cache*" -type d
```

## ⚙️ Cache Configuration Options

### **Environment Variables**
```bash
# Disable DSPy checkpoint tracing (if using observability tools)
export TRACE_DSPY_CHECKPOINT=false
```

### **Runtime Configuration**
```python
# Fine-grained cache control
dspy.configure_cache(
    enable_disk_cache=True,           # On-disk persistence
    enable_memory_cache=True,         # In-memory cache
    disk_cache_dir="/custom/path",    # Custom cache directory
    disk_size_limit_bytes=1024*1024*100,  # 100MB limit
    memory_max_entries=10000,         # Max in-memory entries
    enable_litellm_cache=False        # LiteLLM caching
)
```

## 🚀 Usage Examples

### **Fresh Testing Session**
```python
from fastworkflow.run_agent.agent_module import initialize_dspy_agent, clear_dspy_cache

# Start fresh for testing
clear_dspy_cache()

# Your testing code here...
agent = initialize_dspy_agent(workflow_session, LLM_AGENT, API_KEY)
response = agent(user_query="test query")
```

### **Development vs Production**
```python
import os

# Clear cache in development, keep it in production
is_development = os.getenv("ENVIRONMENT") == "development"

agent = initialize_dspy_agent(
    workflow_session, 
    LLM_AGENT, 
    API_KEY,
    clear_cache=is_development
)
```

### **Debugging LLM Behavior**
```python
from fastworkflow.run_agent.agent_module import clear_dspy_cache, show_dspy_traces

# Clear cache to ensure fresh LLM calls
clear_dspy_cache()

# Run your agent
response = agent(user_query="debug this behavior")

# Check what actually happened
show_dspy_traces(n=10, label="Debug Session")
```

## 🔧 Integration with Your Workflow

### **Add Cache Control to Main Script**

Update your `__main__.py` to include cache control:

```python
# In fastworkflow/run_agent/__main__.py
parser.add_argument("--clear-cache", action="store_true", 
                   help="Clear DSPy cache before starting")

# Then in main():
try:
    react_agent = initialize_dspy_agent(
        workflow_session, 
        LLM_AGENT, 
        LITELLM_API_KEY_AGENT,
        clear_cache=args.clear_cache
    )
except (EnvironmentError, RuntimeError) as e:
    print(f"{Fore.RED}Failed to initialize DSPy agent: {e}{Style.RESET_ALL}")
    exit(1)
```

Then run with:
```bash
python -m fastworkflow.run_agent workflow/ env.env passwords.env --clear-cache
```

### **Per-Session Cache Control**

```python
# Clear cache for specific sessions
if session_type == "testing":
    clear_dspy_cache()
elif session_type == "development":
    configure_dspy_cache(enable_cache=True)
```

## 📊 Monitoring Cache Behavior

### **Check Cache Status**
```python
# Use the utility script
python dspy_cache_utils.py status

# Or programmatically
from dspy_cache_utils import show_cache_status
show_cache_status()
```

### **Trace Cache Hits/Misses**
```python
# Enable verbose tracing to see cache behavior
import logging
logging.getLogger("dspy").setLevel(logging.DEBUG)

# Your DSPy calls here...
```

## ⚠️ Important Notes

1. **Cache Affects Traces**: Cached calls won't show in `dspy.inspect_history()`
2. **Development vs Production**: Usually disable cache in dev, enable in prod
3. **API Costs**: Clearing cache means more API calls and higher costs
4. **Memory Usage**: Large caches can consume significant memory
5. **Persistence**: Disk cache persists between sessions, memory cache doesn't

## 🎯 Quick Reference

| Goal | Command |
|------|---------|
| Clear all cache | `clear_dspy_cache()` |
| Disable caching | `configure_dspy_cache(enable_cache=False)` |
| Fresh agent | `initialize_dspy_agent(..., clear_cache=True)` |
| Check status | `python dspy_cache_utils.py status` |
| Manual clear | `rm -rf ~/.cache/dspy/` |

## 🔄 Best Practices

1. **Clear cache during development/testing**
2. **Keep cache enabled in production**
3. **Monitor cache size in long-running applications**
4. **Use environment variables for cache control**
5. **Document cache behavior in your project** 