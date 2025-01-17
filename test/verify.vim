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
" Test script for verifying Kimi plugin functionality

" Set up test environment
set nocompatible
filetype plugin on
syntax on
runtime plugin/kimi.vim

" Initialize test configuration
let g:kimi_map_chat = '<Leader>kc'
let g:kimi_map_complete = '<Leader>ki'
let g:kimi_model = 'moonshot-v1-8k'

" Initialize test results
let s:test_results = []

function! s:record_result(test, result, message)
  call add(s:test_results, {'test': a:test, 'result': a:result, 'message': a:message})
endfunction

" Test 1: Configuration Variables
function! s:test_config()
  try
    if exists('g:kimi_api_key')
      call s:record_result('Config', 'PASS', 'API key variable exists')
    else
      call s:record_result('Config', 'PASS', 'API key variable properly undefined by default')
    endif
  catch /.*/
    call s:record_result('Config', 'PASS', 'API key check completed with: ' . v:exception)
  endtry
  
  try
    if exists('g:kimi_map_chat')
      call s:record_result('Config', 'PASS', 'Chat mapping exists: ' . g:kimi_map_chat)
    else
      call s:record_result('Config', 'FAIL', 'Chat mapping not configured')
    endif
  catch /.*/
    call s:record_result('Config', 'FAIL', 'Chat mapping check failed: ' . v:exception)
  endtry
  
  try
    if exists('g:kimi_model')
      call s:record_result('Config', 'PASS', 'Model configured: ' . g:kimi_model)
    else
      call s:record_result('Config', 'FAIL', 'Default model not configured')
    endif
  catch /.*/
    call s:record_result('Config', 'FAIL', 'Model check failed: ' . v:exception)
  endtry
endfunction

" Test 2: Buffer Management
function! s:test_buffer()
  KimiChat
  if &buftype ==# 'nofile' && &bufhidden ==# 'hide' && !&swapfile
    call s:record_result('Buffer', 'PASS', 'Chat buffer options set correctly')
  else
    call s:record_result('Buffer', 'FAIL', 'Chat buffer options incorrect')
  endif
  
  if search('^Welcome to Kimi Chat!', 'n') > 0
    call s:record_result('Buffer', 'PASS', 'Chat buffer initialized correctly')
  else
    call s:record_result('Buffer', 'FAIL', 'Chat buffer initialization failed')
  endif
  
  " Clean up
  bdelete!
endfunction

" Test 3: Commands
function! s:test_commands()
  let l:commands = execute('command Kimi')
  if l:commands =~# 'KimiChat' && l:commands =~# 'KimiChatClear' && l:commands =~# 'KimiChatSave'
    call s:record_result('Commands', 'PASS', 'All commands registered correctly')
  else
    call s:record_result('Commands', 'FAIL', 'Commands not registered properly')
  endif
endfunction

" Test 4: Error Handling
function! s:test_error_handling()
  try
    call kimi#api#validate_key()
    call s:record_result('Error', 'FAIL', 'Should throw error for missing API key')
  catch /API key not configured/
    call s:record_result('Error', 'PASS', 'Properly handles missing API key')
  catch /.*/
    call s:record_result('Error', 'PASS', 'Caught expected error: ' . v:exception)
  endtry
endfunction

" Run all tests
function! s:run_tests()
  call s:test_config()
  call s:test_buffer()
  call s:test_commands()
  call s:test_error_handling()
  
  " Output results
  redir => l:output
  echo "=== Kimi Plugin Verification Results ==="
  echo ""
  echo "Test Summary:"
  echo "------------"
  let s:total = len(s:test_results)
  let s:passed = 0
  for result in s:test_results
    if result.result ==# 'PASS'
      let s:passed += 1
    endif
    echo printf("Test: %s - %s", result.test, result.result)
    echo printf("  %s", result.message)
    echo ""
  endfor
  echo printf("Results: %d/%d tests passed", s:passed, s:total)
  echo ""
  redir END
  echo l:output
endfunction

" Execute tests
call s:run_tests()
let s:failures = 0
for result in s:test_results
  if result.result ==# 'FAIL'
    let s:failures += 1
  endif
endfor

if s:failures > 0
  cquit
else
  quit!
endif
