# Vim Kimi Assistant

A Vim plugin that integrates Kimi AI (by Moonshot) into your Vim workflow for AI-powered chat and code completion assistance.

## Features

- AI-powered code completion
- Interactive chat interface within Vim
- Context-aware code suggestions
- Support for multiple Kimi models (8k, 32k, 128k context windows)

## Installation

### Using a Plugin Manager (recommended)

Using [vim-plug](https://github.com/junegunn/vim-plug):

```viml
Plug 'your-username/vim-kimi-assistant'
```

### Manual Installation

```bash
mkdir -p ~/.vim/pack/plugins/start
cd ~/.vim/pack/plugins/start
git clone https://github.com/your-username/vim-kimi-assistant.git
```

For Neovim:
```bash
mkdir -p ~/.local/share/nvim/site/pack/plugins/start
cd ~/.local/share/nvim/site/pack/plugins/start
git clone https://github.com/your-username/vim-kimi-assistant.git
```

## Configuration

1. Get your Kimi API key from [Moonshot Platform](https://platform.moonshot.cn)
2. Add to your .vimrc:
```viml
let g:kimi_api_key = 'your-api-key-here'
```

Optional settings:
```viml
" Key mappings (defaults shown)
let g:kimi_map_chat = '<Leader>kc'     " Open chat window
let g:kimi_map_complete = '<Leader>ki'  " Trigger code completion
```

## Usage

### Chat Interface
1. Press `<Leader>kc` to open the Kimi chat window
2. Type your message
3. Press `<C-]>` to send the message
4. View Kimi's response in the chat window

### Code Completion
1. Select code in visual mode
2. Press `<Leader>ki`
3. Enter your instruction
4. Review and accept/reject the suggested changes

## Requirements

- Vim 8.1+ or Neovim 0.5+
- Python 3.7+
- Internet connection for API access

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
