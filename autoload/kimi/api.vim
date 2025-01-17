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
" kimi/api.vim - Kimi API integration layer
" Maintainer: Moonshot AI
" Version: 0.1

let s:api_base_url = 'https://api.moonshot.cn/v1'
let s:rate_limits = {'rpm': 0, 'tpm': 0, 'last_reset': localtime()}

" Reset rate limits every minute
function! s:reset_rate_limits() abort
  let l:current_time = localtime()
  if l:current_time - s:rate_limits.last_reset >= 60
    let s:rate_limits.rpm = 0
    let s:rate_limits.tpm = 0
    let s:rate_limits.last_reset = l:current_time
  endif
endfunction

" Check if we're within rate limits
function! s:check_rate_limits(estimated_tokens) abort
  call s:reset_rate_limits()
  
  " Default RPM limit is 20, TPM limit is 200k
  if s:rate_limits.rpm >= 20
    throw 'Rate limit exceeded: Too many requests per minute'
  endif
  
  if s:rate_limits.tpm + a:estimated_tokens >= 200000
    throw 'Rate limit exceeded: Too many tokens per minute'
  endif
  
  let s:rate_limits.rpm += 1
  let s:rate_limits.tpm += a:estimated_tokens
endfunction

" Validate API key
function! kimi#api#validate_key() abort
  if !exists('g:kimi_api_key') || empty(g:kimi_api_key)
    throw 'Kimi API key not configured. Please set g:kimi_api_key in your vimrc.'
  endif
  return 1
endfunction

" Prepare API request headers
function! s:prepare_headers() abort
  return {
    \ 'Content-Type': 'application/json',
    \ 'Authorization': 'Bearer ' . g:kimi_api_key
    \ }
endfunction

" Estimate token count for a message
function! s:estimate_tokens(text) abort
  " Rough estimation: 1 token ≈ 4 chars for English, 2 chars for Chinese
  return len(a:text) / 3
endfunction

" Send request to Kimi API
function! kimi#api#request(messages, opts = {}) abort
  call kimi#api#validate_key()
  
  " Estimate tokens for rate limiting
  let l:total_tokens = 0
  for msg in a:messages
    let l:total_tokens += s:estimate_tokens(msg.content)
  endfor
  let l:total_tokens += get(a:opts, 'max_tokens', 1000)
  
  call s:check_rate_limits(l:total_tokens)
  
  let l:data = {
    \ 'model': get(a:opts, 'model', 'moonshot-v1-8k'),
    \ 'messages': a:messages,
    \ 'temperature': get(a:opts, 'temperature', 0.3),
    \ 'stream': get(a:opts, 'stream', v:true),
    \ 'max_tokens': get(a:opts, 'max_tokens', 1000)
    \ }
    
  let l:cmd = ['curl', '-s', '-X', 'POST']
  let l:cmd += ['-H', 'Content-Type: application/json']
  let l:cmd += ['-H', 'Authorization: Bearer ' . g:kimi_api_key]
  let l:cmd += ['-d', json_encode(l:data)]
  let l:cmd += [s:api_base_url . '/chat/completions']
  
  let l:job_opts = {
    \ 'callback': function(get(a:opts, 'callback', 's:default_callback')),
    \ 'on_stderr': function('s:handle_error'),
    \ 'mode': 'raw'
    \ }
    
  if has('nvim')
    let l:job = jobstart(l:cmd, l:job_opts)
  else
    let l:job = job_start(l:cmd, l:job_opts)
  endif
  
  return l:job
endfunction

" Default callback for handling responses
function! s:default_callback(channel, msg) abort
  if type(a:msg) == v:t_string && !empty(a:msg)
    try
      let l:data = json_decode(a:msg)
      if has_key(l:data, 'error')
        throw l:data.error.message
      endif
      echo l:data.choices[0].message.content
    catch
      echohl ErrorMsg
      echo 'Error processing response: ' . v:exception
      echohl None
    endtry
  endif
endfunction

" Handle API errors
function! s:handle_error(channel, msg) abort
  echohl ErrorMsg
  echo 'API Error: ' . a:msg
  echohl None
endfunction

" Chat completion request
function! kimi#api#chat_completion(messages, opts = {}) abort
  return kimi#api#request(a:messages, a:opts)
endfunction

" Code completion request
function! kimi#api#code_completion(code, instruction, opts = {}) abort
  let l:messages = [
    \ {'role': 'system', 'content': 'You are a code completion assistant. Provide concise, focused code modifications.'},
    \ {'role': 'user', 'content': "Code:\n```\n" . a:code . "\n```\n\nInstruction: " . a:instruction}
    \ ]
  return kimi#api#request(l:messages, a:opts)
endfunction
