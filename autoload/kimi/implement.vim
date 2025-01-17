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
" kimi/implement.vim - Kimi code implementation interface
" Maintainer: Moonshot AI
" Version: 0.1

let s:implementation_bufname = 'Kimi Implementation'
let s:original_code = ''
let s:original_filetype = ''
let s:original_winnr = -1
let s:original_bufnr = -1

" Initialize implementation buffer
function! kimi#implement#init() abort
  " Store original window and buffer
  let s:original_winnr = winnr()
  let s:original_bufnr = bufnr('%')
  let s:original_filetype = &filetype
  
  " Get selected text
  let l:selection = s:get_visual_selection()
  let s:original_code = l:selection
  
  " Create new buffer
  execute 'vnew ' . s:implementation_bufname
  setlocal buftype=nofile
  setlocal bufhidden=hide
  setlocal noswapfile
  setlocal nobuflisted
  
  " Set buffer options
  execute 'setlocal filetype=' . s:original_filetype
  setlocal wrap
  setlocal number
  
  " Set buffer-local mappings
  nnoremap <buffer> <CR> :call kimi#implement#accept_changes()<CR>
  nnoremap <buffer> q :call kimi#implement#close()<CR>
  
  " Display original code
  call append(0, split(s:original_code, "\n"))
endfunction

" Get visual selection
function! s:get_visual_selection() abort
  let [l:line_start, l:column_start] = getpos("'<")[1:2]
  let [l:line_end, l:column_end] = getpos("'>")[1:2]
  let l:lines = getline(l:line_start, l:line_end)
  
  if len(l:lines) == 0
    return ''
  endif
  
  let l:lines[-1] = l:lines[-1][:l:column_end - 1]
  let l:lines[0] = l:lines[0][l:column_start - 1:]
  
  return join(l:lines, "\n")
endfunction

" Handle implementation response
function! s:handle_implementation(channel, msg) abort
  if type(a:msg) == v:t_string && !empty(a:msg)
    try
      let l:data = json_decode(a:msg)
      if has_key(l:data, 'choices') && len(l:data.choices) > 0
        let l:content = l:data.choices[0].message.content
        
        " Extract code from markdown code blocks if present
        let l:code = s:extract_code_blocks(l:content)
        if empty(l:code)
          let l:code = l:content
        endif
        
        " Update implementation buffer
        call s:update_implementation_buffer(l:code)
      endif
    catch
      echohl ErrorMsg
      echo 'Error processing implementation: ' . v:exception
      echohl None
    endtry
  endif
endfunction

" Extract code from markdown code blocks
function! s:extract_code_blocks(content) abort
  let l:blocks = []
  let l:lines = split(a:content, "\n")
  let l:in_block = 0
  let l:current_block = []
  
  for l:line in l:lines
    if l:line =~# '^```'
      if l:in_block
        let l:in_block = 0
        if !empty(l:current_block)
          call add(l:blocks, join(l:current_block, "\n"))
        endif
        let l:current_block = []
      else
        let l:in_block = 1
      endif
    elseif l:in_block
      call add(l:current_block, l:line)
    endif
  endfor
  
  return join(l:blocks, "\n\n")
endfunction

" Update implementation buffer
function! s:update_implementation_buffer(code) abort
  let l:bufnr = bufnr(s:implementation_bufname)
  if l:bufnr != -1
    " Clear buffer
    execute l:bufnr . 'bdelete!'
    
    " Create new buffer with updated code
    execute 'vnew ' . s:implementation_bufname
    setlocal buftype=nofile
    setlocal bufhidden=hide
    setlocal noswapfile
    setlocal nobuflisted
    execute 'setlocal filetype=' . s:original_filetype
    
    " Set buffer-local mappings
    nnoremap <buffer> <CR> :call kimi#implement#accept_changes()<CR>
    nnoremap <buffer> q :call kimi#implement#close()<CR>
    
    " Insert new code
    call append(0, split(a:code, "\n"))
    normal! gg
  endif
endfunction

" Accept implementation changes
function! kimi#implement#accept_changes() abort
  let l:new_code = join(getline(1, '$'), "\n")
  
  " Switch back to original window
  execute s:original_winnr . 'wincmd w'
  
  " Replace original selection with new code
  let l:lines = split(l:new_code, "\n")
  execute "'<,'>delete"
  call append("'<-1", l:lines)
  
  " Clean up implementation buffer
  call kimi#implement#close()
endfunction

" Close implementation buffer
function! kimi#implement#close() abort
  let l:bufnr = bufnr(s:implementation_bufname)
  if l:bufnr != -1
    execute l:bufnr . 'bdelete!'
  endif
  
  " Return to original window
  if s:original_winnr != -1
    execute s:original_winnr . 'wincmd w'
  endif
endfunction

" Request code implementation
function! kimi#implement#request_implementation(instruction) abort
  " Prepare system message
  let l:system_msg = 'You are a code implementation assistant. Analyze the code and provide improvements based on the instruction. Return only the modified code without explanations.'
  
  " Prepare messages for API
  let l:messages = [
    \ {'role': 'system', 'content': l:system_msg},
    \ {'role': 'user', 'content': "Code:\n```\n" . s:original_code . "\n```\n\nInstruction: " . a:instruction}
    \ ]
  
  " Send request
  let l:opts = {
    \ 'callback': function('s:handle_implementation'),
    \ 'stream': v:false
    \ }
  call kimi#api#request(l:messages, l:opts)
endfunction

" Complete code implementation
function! kimi#implement#complete() range abort
  call kimi#implement#init()
  let l:instruction = input('Enter completion instruction: ')
  if !empty(l:instruction)
    call kimi#implement#request_implementation(l:instruction)
  endif
endfunction
