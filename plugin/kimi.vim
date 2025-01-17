" Copyright 2025 Moonshot AI
"
" Licensed under the Apache License, Version 2.0 (the "License");
" you may not use this file except in compliance with the License.
" You may obtain a copy of the License at
"
"     http://www.apache.org/licenses/LICENSE-2.0
"
" Unless required by applicable law or agreed to in writing, software
" distributed under the License is distributed on an "AS IS" BASIS,
" WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
" See the License for the specific language governing permissions and
" limitations under the License.
"
" kimi.vim - Kimi AI Assistant for Vim
" Maintainer: Moonshot AI
" Version: 0.1

if exists('g:loaded_kimi')
  finish
endif
let g:loaded_kimi = 1

" Default configuration
let g:kimi_api_key = get(g:, 'kimi_api_key', '')
let g:kimi_map_chat = get(g:, 'kimi_map_chat', '<Leader>kc')
let g:kimi_map_complete = get(g:, 'kimi_map_complete', '<Leader>ki')
let g:kimi_model = get(g:, 'kimi_model', 'moonshot-v1-8k')
let g:kimi_temperature = get(g:, 'kimi_temperature', 0.3)
let g:kimi_max_tokens = get(g:, 'kimi_max_tokens', 1000)

" Validate requirements
if !has('python3')
  echoerr 'Kimi: Python 3 support is required'
  finish
endif

if !has('job')
  echoerr 'Kimi: +job feature is required'
  finish
endif

" Commands
command! -nargs=0 KimiChat call kimi#chat#init()
command! -nargs=0 KimiChatClear call kimi#chat#clear()
command! -nargs=1 -complete=file KimiChatSave call kimi#chat#save(<f-args>)
command! -range -nargs=? KimiImplement call kimi#implement#complete()

" Mappings
execute 'nnoremap ' . g:kimi_map_chat . ' :KimiChat<CR>'
execute 'vnoremap ' . g:kimi_map_complete . ' :KimiImplement<CR>'

" Autocommands
augroup Kimi
  autocmd!
  " Add syntax highlighting for chat buffer
  autocmd FileType * if expand('<amatch>') =~# 'Kimi Chat' | 
    \ syntax match KimiUser /^You: .*/hi |
    \ syntax match KimiAssistant /^Assistant: .*/ |
    \ highlight KimiUser ctermfg=green guifg=#00ff00 |
    \ highlight KimiAssistant ctermfg=blue guifg=#0000ff |
    \ endif
augroup END
