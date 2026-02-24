#!/bin/bash

echo "🚀 Installing Promptomatix..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "✅ uv $(uv --version) detected"

# Check if git submodules are initialized
if [ ! -d "libs/dspy" ]; then
    echo "📦 Initializing git submodules..."
    git submodule update --init --recursive
fi

# Install all dependencies and create virtual environment via uv
echo "📦 Installing dependencies..."
uv sync

# Download NLTK data
echo "📥 Downloading NLTK data..."
uv run python -c "
import nltk
for pkg in ['punkt', 'averaged_perceptron_tagger', 'wordnet']:
    nltk.download(pkg, quiet=True)
print('✅ NLTK data download complete!')
"

echo ""
echo "🎉 Installation complete!"
echo ""
echo "🔧 Usage:"
echo "   uv run promtomatic --help"
echo ""
echo "🔑 Set up your API keys:"
echo "   1. Copy the sample environment file:"
echo "      cp .env.example .env"
echo "   2. Edit .env and add your API keys:"
echo "      nano .env  # or use any text editor"
echo "   3. Make sure to replace 'your_key_here' with your actual API keys"
echo ""
echo " Quick start:"
echo "   1. Set up .env file (see above)"
echo "   2. Test: uv run promtomatic --raw_input 'Classify sentiment'"
echo ""
echo "📚 For more help: uv run promtomatic --help"
