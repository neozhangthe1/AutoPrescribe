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
" kimi/chat.vim - Kimi chat interface
" Maintainer: Moonshot AI
" Version: 0.1

let s:chat_bufname = 'Kimi Chat'
let s:chat_history = []
let s:current_response = ''
let s:is_processing = 0

" Initialize chat buffer
function! kimi#chat#init() abort
  " Create new buffer if it doesn't exist
  let l:bufnr = bufnr(s:chat_bufname)
  if l:bufnr == -1
    execute 'vnew ' . s:chat_bufname
    setlocal buftype=nofile
    setlocal bufhidden=hide
    setlocal noswapfile
    setlocal nobuflisted
    setlocal wrap
    setlocal nonumber
    setlocal norelativenumber
    setlocal nocursorline
    
    " Set buffer-local mappings
    nnoremap <buffer> <CR> :call kimi#chat#send_message()<CR>
    inoremap <buffer> <C-]> <Esc>:call kimi#chat#send_message()<CR>
    
    " Set buffer-local syntax
    syntax match KimiUser /^You: .*/
    syntax match KimiAssistant /^Assistant: .*/
    highlight KimiUser ctermfg=green guifg=#00ff00
    highlight KimiAssistant ctermfg=blue guifg=#0000ff
    
    " Initialize chat
    call append(0, ['Welcome to Kimi Chat!', '', 'Type your message and press <C-]> to send.', ''])
  else
    " Switch to existing buffer
    let l:winnr = bufwinnr(l:bufnr)
    if l:winnr == -1
      execute 'vsplit'
      execute 'buffer' l:bufnr
    else
      execute l:winnr . 'wincmd w'
    endif
  endif
  
  " Move cursor to end of buffer
  normal! G
  startinsert!
endfunction

" Handle streaming response
function! s:handle_stream(channel, msg) abort
  if type(a:msg) == v:t_string && !empty(a:msg)
    try
      let l:lines = split(a:msg, "\n")
      for l:line in l:lines
        if l:line =~# '^data: '
          let l:data = l:line[6:]
          if l:data ==# '[DONE]'
            let s:is_processing = 0
            call s:append_message('')
            return
          endif
          
          let l:json = json_decode(l:data)
          if has_key(l:json, 'choices') && len(l:json.choices) > 0
            let l:content = l:json.choices[0].delta.content
            if !empty(l:content)
              let s:current_response .= l:content
              call s:update_response()
            endif
          endif
        endif
      endfor
    catch
      echohl ErrorMsg
      echo 'Error processing stream: ' . v:exception
      echohl None
      let s:is_processing = 0
    endtry
  endif
endfunction

" Update response in buffer
function! s:update_response() abort
  let l:bufnr = bufnr(s:chat_bufname)
  if l:bufnr != -1
    let l:lines = split(s:current_response, "\n")
    let l:last_line = line('$')
    
    " Update last message
    if getline(l:last_line) =~# '^Assistant: '
      call setline(l:last_line, 'Assistant: ' . l:lines[0])
      if len(l:lines) > 1
        call append(l:last_line, l:lines[1:])
      endif
    else
      call append(l:last_line, 'Assistant: ' . l:lines[0])
      if len(l:lines) > 1
        call append(l:last_line + 1, l:lines[1:])
      endif
    endif
    
    " Scroll to bottom
    normal! G
    redraw
  endif
endfunction

" Append message to buffer
function! s:append_message(msg) abort
  let l:bufnr = bufnr(s:chat_bufname)
  if l:bufnr != -1
    call append(line('$'), a:msg)
    normal! G
    redraw
  endif
endfunction

" Send message to Kimi
function! kimi#chat#send_message() abort
  if s:is_processing
    echo 'Still processing previous message...'
    return
  endif
  
  " Get message from current line
  let l:msg = getline('.')
  if empty(l:msg)
    return
  endif
  
  " Format message
  if l:msg !~# '^You: '
    let l:msg = 'You: ' . l:msg
    call setline('.', l:msg)
  endif
  
  " Add to history
  call add(s:chat_history, {'role': 'user', 'content': l:msg[5:]})
  
  " Prepare messages for API
  let l:messages = [
    \ {'role': 'system', 'content': 'You are kimi.vim, an AI pair programmer focused on helping users with their code. Your responses should be concise and focused on the task.'}
    \ ]
  call extend(l:messages, s:chat_history[-10:])
  
  " Reset response buffer
  let s:current_response = ''
  let s:is_processing = 1
  
  " Send request
  let l:opts = {
    \ 'callback': function('s:handle_stream'),
    \ 'stream': v:true
    \ }
  call kimi#api#chat_completion(l:messages, l:opts)
  
  " Add new line for next message
  call append(line('$'), '')
  normal! G
  startinsert!
endfunction

" Clear chat history
function! kimi#chat#clear() abort
  let s:chat_history = []
  let l:bufnr = bufnr(s:chat_bufname)
  if l:bufnr != -1
    execute 'bdelete! ' . l:bufnr
  endif
  call kimi#chat#init()
endfunction

" Save chat history
function! kimi#chat#save(filename) abort
  let l:content = []
  for msg in s:chat_history
    call add(l:content, msg.role . ': ' . msg.content)
  endfor
  call writefile(l:content, a:filename)
  echo 'Chat history saved to ' . a:filename
endfunction
