// quarkus-llm-chat REST + EventSource SSE client

(function () {
    'use strict';

    // --- Login ---
    var currentUser = localStorage.getItem('llm-chat-user') || '';

    function showLogin() {
        document.getElementById('login-screen').style.display = 'flex';
        document.getElementById('app').style.display = 'none';
        var input = document.getElementById('login-user');
        input.value = currentUser;
        input.focus();
    }

    function startApp(username) {
        currentUser = username;
        localStorage.setItem('llm-chat-user', username);
        document.getElementById('login-screen').style.display = 'none';
        document.getElementById('app').style.display = '';
        document.getElementById('user-label').textContent = username;
        initApp();
    }

    document.getElementById('login-form').addEventListener('submit', function (e) {
        e.preventDefault();
        var username = document.getElementById('login-user').value.trim();
        if (username) {
            startApp(username);
        }
    });

    // Check single-user mode from server config
    fetch('api/config').then(function (r) { return r.json(); }).then(function (cfg) {
        if (cfg.singleUserMode) {
            startApp('default');
        } else if (currentUser) {
            startApp(currentUser);
        } else {
            showLogin();
        }
    }).catch(function () {
        if (currentUser) {
            startApp(currentUser);
        } else {
            showLogin();
        }
    });

    var appInitialized = false;
    function initApp() {
        if (appInitialized) return;
        appInitialized = true;
        doInitApp();
    }

    function apiUrl(path) {
        return path + (path.indexOf('?') >= 0 ? '&' : '?') + 'user=' + encodeURIComponent(currentUser);
    }

    function doInitApp() {

    const chatArea = document.getElementById('chat-area');
    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const queueBtn = document.getElementById('queue-btn');
    const cancelBtn = document.getElementById('cancel-btn');
    const modelSelect = document.getElementById('model-select');
    const themeSelect = document.getElementById('theme-select');
    const sessionLabel = document.getElementById('session-label');
    const connectionStatus = document.getElementById('connection-status');
    const queueArea = document.getElementById('queue-area');
    const activityLabel = document.getElementById('activity-label');
    const inputArea = document.getElementById('input-area');
    const imagePreviewArea = document.getElementById('image-preview-area');
    const imageBtn = document.getElementById('image-btn');
    const imageFileInput = document.getElementById('image-file-input');

    // Pending images as data URLs
    var pendingImages = [];

    // Fetched URL contents: [{url, content}]
    var fetchedUrls = [];

    let thinkingStartTime = null;   // Date.now() when thinking started
    let thinkingTimer = null;       // setInterval ID

    // --- Per-session localStorage isolation ---
    // When served under /session/{id}/ (k8s-pups), suffix localStorage keys with the
    // session ID so multiple instances don't share chat history, prompt queue, theme,
    // and model selection.  When served at / (standalone), suffix is empty — backward compatible.
    var SESSION_SUFFIX = (function () {
        var m = window.location.pathname.match(/^\/session\/([^/]+)\//);
        return m ? '-' + m[1] : '';
    })();

    // --- Theme (per-session in k8s-pups, global in standalone) ---
    var THEME_KEY = 'coder-agent-theme' + SESSION_SUFFIX;
    var savedTheme = localStorage.getItem(THEME_KEY) || 'dark-catppuccin';
    console.log('[coder-agent] theme restore: key=' + THEME_KEY + ' saved=' + localStorage.getItem(THEME_KEY) + ' using=' + savedTheme);
    document.documentElement.setAttribute('data-theme', savedTheme);
    themeSelect.value = savedTheme;

    themeSelect.addEventListener('change', function () {
        var theme = themeSelect.value;
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem(THEME_KEY, theme);
        console.log('[coder-agent] theme saved: key=' + THEME_KEY + ' value=' + theme);
    });

    // --- Model key (per-session in k8s-pups, global in standalone) ---
    var MODEL_KEY = 'coder-agent-model' + SESSION_SUFFIX;

    let currentAssistantMsg = null;
    let currentAssistantText = '';
    let needsParagraphBreak = false; // insert \n\n before next delta (after tool use etc.)
    let busy = false;
    let pendingPrompt = false;  // true when an ask_user prompt is awaiting user response

    // --- Chat history (persisted to localStorage) ---
    var HISTORY_KEY = 'coder-agent-history-' + currentUser + SESSION_SUFFIX;
    var MAX_HISTORY = 500;
    var chatHistory = []; // [{role: 'user'|'assistant'|'info'|'error', text: string}]

    // --- Prompt Queue (position-based, persisted to localStorage) ---
    var QUEUE_KEY = 'coder-agent-queue-' + currentUser + SESSION_SUFFIX;
    var queue = [];   // [{text: string, auto: boolean}]
    var queuePos = 0; // index of next item to send
    var MAX_QUEUE_SIZE = 100;

    // Configure marked for safe rendering
    marked.setOptions({
        breaks: true,
        gfm: true
    });

    // --- Timestamp helper ---

    function formatTime(date) {
        var y = date.getFullYear();
        var m = String(date.getMonth() + 1).padStart(2, '0');
        var d = String(date.getDate()).padStart(2, '0');
        var hh = String(date.getHours()).padStart(2, '0');
        var mm = String(date.getMinutes()).padStart(2, '0');
        var ss = String(date.getSeconds()).padStart(2, '0');
        var tz = -date.getTimezoneOffset();
        var tzSign = tz >= 0 ? '+' : '-';
        var tzH = String(Math.floor(Math.abs(tz) / 60)).padStart(2, '0');
        var tzM = String(Math.abs(tz) % 60).padStart(2, '0');
        return y + '-' + m + '-' + d + 'T' + hh + ':' + mm + ':' + ss + tzSign + tzH + ':' + tzM;
    }

    // --- Model loading ---

    function loadModels() {
        fetch(apiUrl('api/models'))
            .then(function (resp) { return resp.json(); })
            .then(function (models) {
                modelSelect.innerHTML = '';
                // Count distinct servers among local models
                var servers = {};
                models.forEach(function (m) {
                    if (m.type === 'local' && m.server) {
                        servers[m.server] = true;
                    }
                });
                var serverCount = Object.keys(servers).length;

                // Local models first, then Claude models
                var localModels = models.filter(function (m) { return m.type === 'local'; });
                var cloudModels = models.filter(function (m) { return m.type !== 'local'; });
                var sorted = localModels.concat(cloudModels);

                sorted.forEach(function (m) {
                    var opt = document.createElement('option');
                    opt.value = m.name;
                    opt.setAttribute('data-type', m.type);
                    if (m.type === 'local') {
                        opt.textContent = serverCount > 1 && m.server
                            ? m.name + ' (' + m.server + ')'
                            : m.name + ' (local)';
                    } else {
                        opt.textContent = m.name;
                    }
                    modelSelect.appendChild(opt);
                });

                // Restore previously selected model from localStorage
                var savedModel = localStorage.getItem(MODEL_KEY);
                console.log('[coder-agent] model restore: saved=' + savedModel + ' options=' + modelSelect.options.length);
                if (savedModel) {
                    var found = false;
                    for (var i = 0; i < modelSelect.options.length; i++) {
                        if (modelSelect.options[i].value === savedModel) {
                            modelSelect.value = savedModel;
                            found = true;
                            break;
                        }
                    }
                    console.log('[coder-agent] model restore: found=' + found + ' current=' + modelSelect.value);
                }
            })
            .catch(function () {
                modelSelect.innerHTML = '';
                var opt = document.createElement('option');
                opt.value = '';
                opt.textContent = '(no models available)';
                opt.disabled = true;
                modelSelect.appendChild(opt);
            });
    }

    // --- EventSource SSE connection ---

    var eventSource = null;
    var reconnectDebounce = null;
    var reconnectAttempts = 0;

    function connectSSE() {
        if (eventSource) {
            eventSource.close();
        }
        eventSource = new EventSource(apiUrl('api/chat/stream'));

        eventSource.onopen = function () {
            // Clear any pending reconnecting indicator
            if (reconnectDebounce) {
                clearTimeout(reconnectDebounce);
                reconnectDebounce = null;
            }
            reconnectAttempts = 0;
            connectionStatus.textContent = 'ready';
            connectionStatus.className = 'connected';
        };

        eventSource.onmessage = function (event) {
            try {
                handleEvent(JSON.parse(event.data));
            } catch (e) {
                // skip non-JSON
            }
        };

        eventSource.onerror = function () {
            // Debounce: only show "reconnecting" if disconnected for >2 seconds
            if (!reconnectDebounce) {
                reconnectDebounce = setTimeout(function () {
                    reconnectDebounce = null;
                    if (!eventSource || eventSource.readyState !== 1) {
                        connectionStatus.textContent = 'reconnecting';
                        connectionStatus.className = 'disconnected';
                    }
                }, 2000);
            }
            // If EventSource gave up (CLOSED), manually reconnect with backoff
            if (eventSource.readyState === 2) {
                reconnectAttempts++;
                var delay = Math.min(reconnectAttempts * 5000, 30000);
                console.log('[SSE] Connection closed, reconnecting in ' + (delay/1000) + 's');
                setTimeout(connectSSE, delay);
            }
        };
    }

    // --- Event handling ---

    function handleEvent(event) {
        switch (event.type) {
            case 'delta':
                handleDelta(event.content);
                break;
            case 'thinking':
                handleThinking(event.content);
                break;
            case 'result':
                handleResult(event);
                break;
            case 'error':
                appendMessage('error', event.content);
                break;
            case 'info':
                appendMessage('info', event.content);
                break;
            case 'status':
                updateStatus(event);
                break;
            case 'prompt':
                handlePrompt(event);
                break;
            case 'log':
                break;
        }
    }

    function startThinkingTimer() {
        if (thinkingTimer) return; // already running
        thinkingStartTime = Date.now();
        thinkingTimer = setInterval(updateElapsed, 1000);
    }

    function stopThinkingTimer() {
        if (thinkingTimer) {
            clearInterval(thinkingTimer);
            thinkingTimer = null;
        }
        thinkingStartTime = null;
        activityLabel.textContent = '';
        activityLabel.removeAttribute('data-base');
    }

    function updateElapsed() {
        if (!thinkingStartTime) return;
        var elapsed = Math.floor((Date.now() - thinkingStartTime) / 1000);
        var suffix = ' (' + elapsed + 's)';
        // Update thinking indicator in chat area
        if (currentAssistantMsg) {
            var indicator = currentAssistantMsg.querySelector('.thinking-indicator');
            if (indicator) {
                var base = indicator.getAttribute('data-base') || indicator.textContent;
                if (!indicator.getAttribute('data-base')) indicator.setAttribute('data-base', base);
                indicator.textContent = base + suffix;
            }
        }
        // Update activity label in status bar
        var base = activityLabel.getAttribute('data-base');
        if (base) {
            activityLabel.textContent = base + suffix;
        }
    }

    function handleThinking(content) {
        // Re-enable cancel if server is still active (e.g., after POST timeout)
        if (!busy) {
            busy = true;
            cancelBtn.disabled = false;
        }
        if (!currentAssistantMsg) {
            currentAssistantMsg = document.createElement('div');
            currentAssistantMsg.className = 'message assistant streaming';
            chatArea.appendChild(currentAssistantMsg);
        }
        if (!currentAssistantText) {
            // Show thinking indicator before any text has arrived
            var indicator = document.createElement('div');
            indicator.className = 'thinking-indicator';
            indicator.textContent = content || 'Thinking...';
            currentAssistantMsg.appendChild(indicator);
            scrollToBottom();
        } else {
            // Text already accumulated; insert paragraph break before next delta
            needsParagraphBreak = true;
        }
        // Update activity label in status bar
        var label = content || 'Thinking...';
        activityLabel.setAttribute('data-base', label);
        activityLabel.textContent = label;
        startThinkingTimer();
    }

    function handleDelta(content) {
        if (!content) return;

        // Re-enable cancel if server is still active (e.g., after POST timeout)
        if (!busy) {
            busy = true;
            cancelBtn.disabled = false;
        }

        if (!currentAssistantMsg) {
            currentAssistantMsg = document.createElement('div');
            currentAssistantMsg.className = 'message assistant streaming';
            chatArea.appendChild(currentAssistantMsg);
        }

        if (needsParagraphBreak) {
            needsParagraphBreak = false;
            if (currentAssistantText && !currentAssistantText.endsWith('\n\n')) {
                currentAssistantText += '\n\n';
            }
        }

        currentAssistantText += content;
        // Strip <think>...</think> blocks (Qwen3 thinking mode) from display
        var displayText = currentAssistantText
            .replace(/<think>[\s\S]*?<\/think>/g, '')
            .replace(/<think>[\s\S]*$/, '');  // partial unclosed <think> block
        currentAssistantMsg.innerHTML = marked.parse(displayText);
        scrollToBottom();
    }

    function handleResult(msg) {
        stopThinkingTimer();
        if (currentAssistantMsg) {
            currentAssistantMsg.classList.remove('streaming');

            var footer = document.createElement('div');
            footer.className = 'message-footer';


            if (msg.costUsd != null && msg.costUsd > 0) {
                var cost = document.createElement('span');
                cost.textContent = 'Cost: $' + msg.costUsd.toFixed(4);
                footer.appendChild(cost);
            }
            if (msg.durationMs != null && msg.durationMs >= 0) {
                var duration = document.createElement('span');
                var secs = (msg.durationMs / 1000).toFixed(1);
                duration.textContent = 'Duration: ' + secs + 's';
                footer.appendChild(duration);
            }
            if (msg.sessionId) {
                var session = document.createElement('span');
                session.textContent = 'Session: ' + msg.sessionId.substring(0, 12) + '...';
                session.title = msg.sessionId;
                footer.appendChild(session);
            }

            // Model name
            var modelName = modelSelect.value || '';
            if (modelName) {
                var modelSpan = document.createElement('span');
                modelSpan.title = modelName;
                modelSpan.textContent = modelName.length > 30
                    ? modelName.substring(0, 30) + '...' : modelName;
                footer.appendChild(modelSpan);
            }

            // Copy as Markdown button
            var copyBtn = document.createElement('button');
            copyBtn.className = 'copy-md-btn';
            copyBtn.textContent = 'Copy MD';
            copyBtn.title = 'Copy as Markdown';
            var mdText = currentAssistantText;
            copyBtn.addEventListener('click', function () {
                navigator.clipboard.writeText(mdText).then(function () {
                    copyBtn.textContent = 'Copied!';
                    setTimeout(function () { copyBtn.textContent = 'Copy MD'; }, 1500);
                });
            });
            footer.appendChild(copyBtn);

            var timeSpan = document.createElement('span');
            timeSpan.textContent = formatTime(new Date());
            footer.appendChild(timeSpan);

            currentAssistantMsg.appendChild(footer);

            currentAssistantMsg = null;
            chatHistory.push({ role: 'assistant', text: currentAssistantText });
            currentAssistantText = '';
            needsParagraphBreak = false;
            saveHistory();
            scrollToBottom();
            trimChatArea();
        }

        // result event is the authoritative signal that a turn is complete.
        // Update status from result event (model, session, busy) and process queue.
        if (msg.model) {
            updateStatus(msg);
        }
        if (msg.busy === false) {
            busy = false;
            cancelBtn.disabled = true;
            promptInput.focus();
            processQueue();
        }
    }

    function handlePrompt(event) {
        var div = document.createElement('div');
        div.className = 'message prompt';
        var contentP = document.createElement('p');
        contentP.textContent = event.content || 'Prompt from Claude';
        div.appendChild(contentP);

        // All prompts (ask_user, permission, etc.) are sent via /api/respond
        // which writes to Claude CLI's stdin. The CLI process is still running
        // and waiting for the response in its read loop.

        if (event.options && event.options.length > 0) {
            var btnGroup = document.createElement('div');
            btnGroup.className = 'prompt-buttons';
            event.options.forEach(function (opt) {
                var btn = document.createElement('button');
                btn.className = 'prompt-option-btn';
                btn.textContent = opt;
                btn.addEventListener('click', function () {
                    pendingPrompt = false;
                    sendResponse(event.promptId, opt);
                    div.classList.add('answered');
                    btnGroup.querySelectorAll('button').forEach(
                        function (b) { b.disabled = true; });
                });
                btnGroup.appendChild(btn);
            });
            div.appendChild(btnGroup);
        } else {
            // Free-text input for the response
            var inputRow = document.createElement('div');
            inputRow.className = 'prompt-input-row';
            var input = document.createElement('input');
            input.type = 'text';
            input.className = 'prompt-text-input';
            input.placeholder = 'Type your response...';
            var submitBtn = document.createElement('button');
            submitBtn.className = 'prompt-option-btn';
            submitBtn.textContent = 'Send';
            submitBtn.addEventListener('click', function () {
                if (input.value.trim()) {
                    pendingPrompt = false;
                    sendResponse(event.promptId, input.value.trim());
                    div.classList.add('answered');
                    input.disabled = true;
                    submitBtn.disabled = true;
                }
            });
            input.addEventListener('keydown', function (e) {
                if (e.key === 'Enter' && !e.isComposing) {
                    e.preventDefault();
                    submitBtn.click();
                }
            });
            inputRow.appendChild(input);
            inputRow.appendChild(submitBtn);
            div.appendChild(inputRow);
        }

        chatArea.appendChild(div);
        chatHistory.push({ role: 'prompt', text: event.content || '' });
        saveHistory();
        scrollToBottom();
    }

    // Send text as a regular chat message (for ask_user prompt responses)
    function sendPromptText(text) {
        queue.splice(queuePos, 0, { text: text, auto: true });
        trimQueue();
        renderQueue();
        saveQueue();
        if (!busy) {
            processQueue();
        } else {
            queueArea.style.display = 'block';
        }
    }

    async function sendResponse(promptId, response) {
        try {
            await fetch('api/respond', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ promptId: promptId, response: response })
            });
            chatHistory.push({ role: 'user', text: '[Response] ' + response });
            saveHistory();
        } catch (e) {
            appendMessage('error', 'Failed to send response: ' + e.message);
        }
    }

    function updateStatus(msg) {
        if (msg.model) {
            // Restore from localStorage first; fall back to server-side model
            var savedModel = localStorage.getItem(MODEL_KEY);
            var targetModel = savedModel || msg.model;
            var currentOption = modelSelect.options[modelSelect.selectedIndex];
            var currentIsLocal = currentOption && currentOption.getAttribute('data-type') === 'local';
            console.log('[coder-agent] updateStatus model: server=' + msg.model + ' saved=' + savedModel + ' target=' + targetModel + ' currentIsLocal=' + currentIsLocal);
            if (!currentIsLocal) {
                // Only set if the target model exists in the dropdown
                for (var i = 0; i < modelSelect.options.length; i++) {
                    if (modelSelect.options[i].value === targetModel) {
                        modelSelect.value = targetModel;
                        break;
                    }
                }
            }
        }
        if (msg.sessionId) {
            sessionLabel.textContent = 'Session: ' + msg.sessionId.substring(0, 12) + '...';
            sessionLabel.title = msg.sessionId;
        } else {
            sessionLabel.textContent = '';
        }
        if (msg.busy != null) {
            busy = msg.busy;
            cancelBtn.disabled = !busy;
            if (!busy) {
                stopThinkingTimer();
                promptInput.focus();
                // Safety net: if result event doesn't arrive (e.g., error path),
                // process queue after a longer delay to avoid being stuck.
                // Normal path: handleResult() calls processQueue() immediately.
                setTimeout(function () {
                    if (!busy) processQueue();
                }, 2000);
            }
        }
    }

    var notificationBar = document.getElementById('notification-bar');
    function isTaskNotification(text) {
        return text && text.indexOf('<task-notification>') !== -1;
    }
    function showNotification(text) {
        var item = document.createElement('div');
        item.className = 'notification-item';
        var statusMatch = text.match(/<status>(.*?)<\/status>/);
        var summaryMatch = text.match(/<summary>(.*?)<\/summary>/);
        var status = statusMatch ? statusMatch[1] : 'unknown';
        var summary = summaryMatch ? summaryMatch[1] : 'Background task notification';
        item.textContent = '[' + status + '] ' + summary;
        item.className += ' notification-' + status;
        notificationBar.appendChild(item);
        notificationBar.style.display = 'block';
        setTimeout(function () {
            item.classList.add('notification-fade');
            setTimeout(function () {
                if (item.parentNode) item.parentNode.removeChild(item);
                if (notificationBar.children.length === 0) notificationBar.style.display = 'none';
            }, 500);
        }, 10000);
    }

    function appendMessage(className, text, images) {
        if (className === 'user' && isTaskNotification(text)) {
            showNotification(text);
            return;
        }
        var div = document.createElement('div');
        div.className = 'message ' + className;
        div.textContent = text;
        if (className === 'user') {
            if (images && images.length > 0) {
                var imgDiv = document.createElement('div');
                imgDiv.className = 'user-images';
                images.forEach(function (dataUrl) {
                    var img = document.createElement('img');
                    img.src = dataUrl;
                    img.addEventListener('click', function () {
                        window.open(dataUrl, '_blank');
                    });
                    imgDiv.appendChild(img);
                });
                div.appendChild(imgDiv);
            }
            var footer = document.createElement('div');
            footer.className = 'message-footer';
            var copyBtn = document.createElement('button');
            copyBtn.className = 'copy-md-btn';
            copyBtn.textContent = 'Copy';
            copyBtn.title = 'Copy prompt text';
            copyBtn.addEventListener('click', function () {
                navigator.clipboard.writeText(text).then(function () {
                    copyBtn.textContent = 'Copied!';
                    setTimeout(function () { copyBtn.textContent = 'Copy'; }, 1500);
                });
            });
            footer.appendChild(copyBtn);
            var timeSpan = document.createElement('span');
            timeSpan.textContent = formatTime(new Date());
            footer.appendChild(timeSpan);
            div.appendChild(footer);
        }
        chatArea.appendChild(div);
        chatHistory.push({ role: className, text: text });
        saveHistory();
        scrollToBottom();
        trimChatArea();
    }

    // --- History persistence (localStorage) ---

    function saveHistory() {
        var toSave = chatHistory.slice(-MAX_HISTORY);
        try {
            localStorage.setItem(HISTORY_KEY, JSON.stringify(toSave));
        } catch (e) {
            // localStorage full — trim aggressively
            try {
                localStorage.setItem(HISTORY_KEY, JSON.stringify(chatHistory.slice(-50)));
            } catch (e2) { /* give up */ }
        }
    }

    function restoreHistory() {
        try {
            var saved = localStorage.getItem(HISTORY_KEY);
            if (!saved) return;
            var entries = JSON.parse(saved);
            if (!Array.isArray(entries) || entries.length === 0) return;

            chatHistory = entries.filter(function (e) { return !(e.role === 'user' && isTaskNotification(e.text)); });
            for (var i = 0; i < entries.length; i++) {
                var entry = entries[i];
                if (entry.role === 'assistant') {
                    chatArea.appendChild(createAssistantDiv(entry.text));
                } else if (entry.role === 'prompt') {
                    var div = document.createElement('div');
                    div.className = 'message prompt answered';
                    var p = document.createElement('p');
                    p.textContent = entry.text;
                    div.appendChild(p);
                    chatArea.appendChild(div);
                } else {
                    var div = document.createElement('div');
                    div.className = 'message ' + entry.role;
                    div.textContent = entry.text;
                    chatArea.appendChild(div);
                }
            }
            scrollToBottom();
        } catch (e) {
            // ignore restore errors
        }
    }

    // --- Queue persistence (localStorage) ---

    function saveQueue() {
        try {
            localStorage.setItem(QUEUE_KEY, JSON.stringify({ queue: queue, pos: queuePos }));
        } catch (e) { /* ignore */ }
    }

    function restoreQueue() {
        try {
            var saved = localStorage.getItem(QUEUE_KEY);
            if (!saved) return;
            var data = JSON.parse(saved);
            if (data && Array.isArray(data.queue)) {
                queue = data.queue;
                queuePos = typeof data.pos === 'number' ? data.pos : 0;
                if (queuePos > queue.length) queuePos = queue.length;
                if (queue.length > 0) {
                    queueArea.style.display = 'block';
                    renderQueue();
                }
            }
        } catch (e) { /* ignore */ }
    }

    function createAssistantDiv(text) {
        var div = document.createElement('div');
        div.className = 'message assistant';
        div.innerHTML = marked.parse(text);

        var footer = document.createElement('div');
        footer.className = 'message-footer';

        var copyBtn = document.createElement('button');
        copyBtn.className = 'copy-md-btn';
        copyBtn.textContent = 'Copy MD';
        copyBtn.title = 'Copy as Markdown';
        copyBtn.addEventListener('click', function () {
            navigator.clipboard.writeText(text).then(function () {
                copyBtn.textContent = 'Copied!';
                setTimeout(function () { copyBtn.textContent = 'Copy MD'; }, 1500);
            });
        });
        footer.appendChild(copyBtn);
        div.appendChild(footer);
        return div;
    }

    var userScrolledUp = false;

    chatArea.addEventListener('scroll', function () {
        var threshold = 80;
        var atBottom = chatArea.scrollHeight - chatArea.scrollTop - chatArea.clientHeight < threshold;
        userScrolledUp = !atBottom;
    });

    function scrollToBottom() {
        if (!userScrolledUp) {
            chatArea.scrollTop = chatArea.scrollHeight;
        }
    }

    function forceScrollToBottom() {
        userScrolledUp = false;
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    // --- Chat area memory limit ---
    var MAX_CHAT_LINES = 5000;

    function trimChatArea() {
        if (chatArea.children.length < 3) return;
        var totalLines = chatArea.innerText.split('\n').length;
        while (totalLines > MAX_CHAT_LINES && chatArea.children.length > 1) {
            var oldest = chatArea.children[0];
            var oldestLines = (oldest.innerText || '').split('\n').length;
            chatArea.removeChild(oldest);
            totalLines -= oldestLines;
        }
    }

    // --- Queue management (position-based) ---

    function addToQueue(text) {
        queue.push({ text: text, auto: true });
        trimQueue();
        renderQueue();
        saveQueue();
    }

    function trimQueue() {
        while (queue.length > MAX_QUEUE_SIZE) {
            queue.shift();
            if (queuePos > 0) queuePos--;
        }
    }

    function removeFromQueue(index) {
        queue.splice(index, 1);
        if (index < queuePos) {
            queuePos--;
        } else if (index === queuePos && queuePos >= queue.length) {
            // pos was pointing at removed item which was last
        }
        renderQueue();
        saveQueue();
    }

    function toggleAutoInQueue(index) {
        if (index >= 0 && index < queue.length) {
            queue[index].auto = !queue[index].auto;
            renderQueue();
            saveQueue();
        }
    }

    function moveInQueue(index, direction) {
        var target = index + direction;
        if (index < queuePos || target < queuePos || target >= queue.length) return;
        var tmp = queue[index];
        queue[index] = queue[target];
        queue[target] = tmp;
        renderQueue();
        saveQueue();
    }

    function hasPending() {
        return queuePos < queue.length;
    }

    function renderQueue() {
        if (queue.length === 0) {
            queueArea.innerHTML = '<div class="queue-header"><span>Queue is empty</span></div>';
            return;
        }

        var pending = queue.length - queuePos;
        var headerText = 'Queue (' + queue.length + ')';
        if (pending > 0) {
            headerText += ' - ' + pending + ' pending';
        }
        headerText += ':';

        var html = '<div class="queue-header">'
            + '<span>' + escapeHtml(headerText) + '</span>'
            + '<button class="queue-save-btn" id="queue-save-btn" title="Save as Markdown">Save</button>'
            + '</div>';

        for (var i = 0; i < queue.length; i++) {
            var item = queue[i];
            var displayText = item.text;
            var sent = (i < queuePos);
            var isCurrent = (i === queuePos);
            var isWaiting = (isCurrent && !busy && !item.auto);

            var cls = 'queue-item';
            if (sent) cls += ' sent';
            if (isCurrent) cls += ' current';
            if (isWaiting) cls += ' waiting';

            var checked = item.auto ? ' checked' : '';

            html += '<div class="' + cls + '" data-index="' + i + '">'
                + '<span class="queue-index">' + (i + 1) + '.</span>'
                + '<span class="queue-text" title="' + escapeAttr(item.text) + '">'
                + escapeHtml(displayText) + '</span>';

            if (!sent) {
                html += '<label class="queue-auto">'
                    + '<input type="checkbox"' + checked + ' data-queue-auto="' + i + '"> Auto</label>';
                var canUp = (i > queuePos);
                var canDown = (i < queue.length - 1);
                html += '<button class="queue-move" data-queue-up="' + i + '"'
                    + (canUp ? '' : ' disabled') + ' title="Move up">&uarr;</button>';
                html += '<button class="queue-move" data-queue-down="' + i + '"'
                    + (canDown ? '' : ' disabled') + ' title="Move down">&darr;</button>';
            }

            html += '<button class="queue-remove" data-queue-remove="' + i + '" title="Remove">&times;</button>'
                + '</div>';
        }

        queueArea.innerHTML = html;
        queueArea.scrollTop = queueArea.scrollHeight;
    }

    // Delegate click events on queue area
    queueArea.addEventListener('click', function (e) {
        var removeBtn = e.target.closest('[data-queue-remove]');
        if (removeBtn) {
            var idx = parseInt(removeBtn.getAttribute('data-queue-remove'), 10);
            removeFromQueue(idx);
            return;
        }

        var upBtn = e.target.closest('[data-queue-up]');
        if (upBtn) {
            var idx = parseInt(upBtn.getAttribute('data-queue-up'), 10);
            moveInQueue(idx, -1);
            return;
        }

        var downBtn = e.target.closest('[data-queue-down]');
        if (downBtn) {
            var idx = parseInt(downBtn.getAttribute('data-queue-down'), 10);
            moveInQueue(idx, 1);
            return;
        }

        if (e.target.id === 'queue-save-btn' || e.target.closest('#queue-save-btn')) {
            saveQueueAsMarkdown();
            return;
        }
    });

    queueArea.addEventListener('change', function (e) {
        if (e.target.hasAttribute('data-queue-auto')) {
            var idx = parseInt(e.target.getAttribute('data-queue-auto'), 10);
            toggleAutoInQueue(idx);
        }
    });

    function saveQueueAsMarkdown() {
        if (queue.length === 0) return;

        var lines = ['# Prompt Queue', ''];
        for (var i = 0; i < queue.length; i++) {
            var item = queue[i];
            var marker = (i < queuePos) ? '[x]' : '[ ]';
            lines.push((i + 1) + '. ' + marker + ' ' + item.text);
        }
        lines.push('');

        var blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        var now = new Date();
        var ts = now.getFullYear()
            + String(now.getMonth() + 1).padStart(2, '0')
            + String(now.getDate()).padStart(2, '0')
            + '-' + String(now.getHours()).padStart(2, '0')
            + String(now.getMinutes()).padStart(2, '0');
        a.download = 'prompt-queue-' + ts + '.md';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function saveChatAsMarkdown() {
        if (chatHistory.length === 0) return;

        var now = new Date();
        var dateStr = now.getFullYear() + '-'
            + String(now.getMonth() + 1).padStart(2, '0') + '-'
            + String(now.getDate()).padStart(2, '0') + ' '
            + String(now.getHours()).padStart(2, '0') + ':'
            + String(now.getMinutes()).padStart(2, '0');

        var lines = ['# Conversation - ' + dateStr, ''];

        for (var i = 0; i < chatHistory.length; i++) {
            var entry = chatHistory[i];
            if (entry.role === 'user') {
                lines.push('## User', '', entry.text, '');
            } else if (entry.role === 'assistant') {
                lines.push('## Assistant', '', entry.text, '');
            } else if (entry.role === 'info') {
                lines.push('> [info] ' + entry.text, '');
            } else if (entry.role === 'error') {
                lines.push('> [error] ' + entry.text, '');
            } else if (entry.role === 'prompt') {
                lines.push('> [prompt] ' + entry.text, '');
            }
        }

        var blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        var ts = now.getFullYear()
            + String(now.getMonth() + 1).padStart(2, '0')
            + String(now.getDate()).padStart(2, '0')
            + String(now.getHours()).padStart(2, '0')
            + String(now.getMinutes()).padStart(2, '0');
        a.download = 'conversation-' + ts + '.md';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function escapeHtml(str) {
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function escapeAttr(str) {
        return str.replace(/&/g, '&amp;').replace(/"/g, '&quot;')
                  .replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function processQueue() {
        if (pendingPrompt) return;
        if (!hasPending()) return;

        if (queue[queuePos].auto) {
            sendFromQueue();
        } else {
            renderQueue();
        }
    }

    function sendFromQueue() {
        if (!hasPending()) return;
        var item = queue[queuePos];
        queuePos++;
        renderQueue();
        saveQueue();
        executePrompt(item.text);
    }

    // --- Send prompt ---

    function sendPrompt() {
        var text = promptInput.value.trim();

        if (!text) {
            if (!busy && hasPending() && !queue[queuePos].auto) {
                sendFromQueue();
            }
            return;
        }

        promptInput.value = '';
        autoResize();

        // Slash commands via REST (always immediate, never queued)
        if (text.startsWith('/')) {
            executeSlashCommand(text);
            return;
        }

        // Always add to queue, then send if not busy
        addToQueue(text);
        if (!busy) {
            processQueue();
        } else {
            renderQueue();
            queueArea.style.display = 'block';
        }
    }

    async function executeSlashCommand(text) {
        try {
            var resp = await fetch('api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            var events = await resp.json();
            events.forEach(handleEvent);
        } catch (e) {
            appendMessage('error', 'Command failed: ' + e.message);
        }
    }

    async function executePrompt(text, images) {
        // Capture and clear pending images
        var sendImages = images || pendingImages.slice();
        clearPendingImages();

        // Prepend fetched URL contents before user's question (RAG pattern: context first, question last)
        var sendText = text;
        if (fetchedUrls.length > 0) {
            var urlParts = fetchedUrls.map(function (f) {
                return '<reference url="' + f.url + '">\n' + f.content + '\n</reference>';
            });
            sendText = urlParts.join('\n\n') + '\n\n---\n\n' + text;
            fetchedUrls = [];
            renderFetchedUrls();
        }

        // Display user message with images (show original text, not the appended content)
        appendMessage('user', text, sendImages);
        busy = true;
        cancelBtn.disabled = false;

        // Show immediate thinking indicator (before API responds)
        currentAssistantMsg = document.createElement('div');
        currentAssistantMsg.className = 'message assistant streaming';
        currentAssistantMsg.innerHTML = '<span class="thinking-indicator">Waiting for response...</span>';
        chatArea.appendChild(currentAssistantMsg);
        forceScrollToBottom();
        activityLabel.setAttribute('data-base', 'Waiting for response...');
        activityLabel.textContent = 'Waiting for response...';
        startThinkingTimer();

        // Submit prompt via POST; events arrive through EventSource
        try {
            var payload = { text: sendText, model: modelSelect.value, noThink: !document.getElementById('think-toggle').checked };
            if (sendImages.length > 0) {
                payload.images = sendImages;
            }
            var response = await fetch(apiUrl('api/chat'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) {
                stopThinkingTimer();
                if (currentAssistantMsg && !currentAssistantText) {
                    currentAssistantMsg.remove();
                    currentAssistantMsg = null;
                }
                var errText = await response.text();
                appendMessage('error', 'HTTP ' + response.status + ': ' + errText.substring(0, 200));
                busy = false;
                cancelBtn.disabled = true;
                processQueue();
                return;
            }
            var result = await response.json();
            if (result.type === 'error') {
                appendMessage('error', result.content);
                busy = false;
                cancelBtn.disabled = true;
                processQueue();
            }
            // Events will arrive through the EventSource connection
        } catch (e) {
            // Clean up thinking indicator if no content was streamed
            stopThinkingTimer();
            if (currentAssistantMsg && !currentAssistantText) {
                currentAssistantMsg.remove();
                currentAssistantMsg = null;
            }
            appendMessage('error', 'Request failed: ' + e.message);
            busy = false;
            cancelBtn.disabled = true;
            processQueue();
        }
    }

    async function cancelRequest() {
        try {
            await fetch(apiUrl('api/cancel'), { method: 'POST' });
        } catch (e) {
            // ignore
        }
        appendMessage('info', 'Cancelled');
    }

    // --- Prediction popup ---

    var predictPopup = document.getElementById('predict-popup');
    var predictSelectedIdx = -1;
    var predictCandidates = [];
    var predictFetching = false;

    function showPredictPopup(candidates) {
        predictCandidates = candidates;
        predictSelectedIdx = -1;
        predictPopup.innerHTML = '';
        candidates.forEach(function (text, idx) {
            var item = document.createElement('div');
            item.className = 'predict-item';
            var key = document.createElement('span');
            key.className = 'predict-key';
            key.textContent = (idx + 1);
            var span = document.createElement('span');
            span.className = 'predict-text';
            span.textContent = text;
            item.appendChild(key);
            item.appendChild(span);
            item.addEventListener('click', function () {
                acceptPrediction(idx);
            });
            predictPopup.appendChild(item);
        });
        predictPopup.style.display = 'block';
    }

    function hidePredictPopup() {
        predictPopup.style.display = 'none';
        predictCandidates = [];
        predictSelectedIdx = -1;
        predictFetching = false;
    }

    function acceptPrediction(idx) {
        if (idx < 0 || idx >= predictCandidates.length) return;
        var text = predictCandidates[idx];
        var curVal = promptInput.value;
        // Add a space before appending if the current text doesn't end with whitespace
        var separator = (curVal.length > 0 && !/\s$/.test(curVal)) ? '' : '';
        promptInput.value = curVal + separator + text;
        autoResize();
        hidePredictPopup();
        promptInput.focus();
    }

    function updatePredictSelection(idx) {
        var items = predictPopup.querySelectorAll('.predict-item');
        items.forEach(function (el, i) {
            el.classList.toggle('selected', i === idx);
        });
        predictSelectedIdx = idx;
    }

    function buildPredictContext() {
        // Build context from recent conversation history (last 10 turns)
        var recent = chatHistory.slice(-10);
        var parts = [];
        for (var i = 0; i < recent.length; i++) {
            var entry = recent[i];
            if (entry.role === 'user') {
                parts.push('User: ' + entry.text);
            } else if (entry.role === 'assistant') {
                // Truncate long assistant responses
                var t = entry.text.length > 500
                    ? entry.text.substring(0, 500) + '...' : entry.text;
                parts.push('Assistant: ' + t);
            }
        }
        return parts.join('\n');
    }

    async function fetchPredictions() {
        var text = promptInput.value.trim();
        if (!text) return;
        predictFetching = true;
        predictPopup.innerHTML = '<div class="predict-loading">Predicting...</div>';
        predictPopup.style.display = 'block';

        try {
            var resp = await fetch('api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    history: buildPredictContext(),
                    n: 5
                })
            });
            var data = await resp.json();
            if (data.candidates && data.candidates.length > 0) {
                showPredictPopup(data.candidates);
            } else {
                predictPopup.innerHTML = '<div class="predict-loading">No predictions</div>';
                setTimeout(hidePredictPopup, 1500);
            }
        } catch (err) {
            predictPopup.innerHTML = '<div class="predict-loading">Prediction failed</div>';
            setTimeout(hidePredictPopup, 1500);
        }
        predictFetching = false;
    }

    // --- Input handling (IME-safe) ---

    promptInput.addEventListener('keydown', function (e) {
        // Prediction popup navigation
        if (predictPopup.style.display !== 'none' && predictCandidates.length > 0) {
            if (e.key === 'Escape') {
                e.preventDefault();
                hidePredictPopup();
                return;
            }
            if (e.key === 'ArrowDown' || (e.key === 'Tab' && !e.shiftKey)) {
                e.preventDefault();
                var next = (predictSelectedIdx + 1) % predictCandidates.length;
                updatePredictSelection(next);
                return;
            }
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                var prev = predictSelectedIdx <= 0 ? predictCandidates.length - 1 : predictSelectedIdx - 1;
                updatePredictSelection(prev);
                return;
            }
            if (e.key === 'Enter' && !e.shiftKey && predictSelectedIdx >= 0) {
                e.preventDefault();
                acceptPrediction(predictSelectedIdx);
                return;
            }
            // Number keys 1-9 to select directly
            if (e.key >= '1' && e.key <= '9') {
                var num = parseInt(e.key) - 1;
                if (num < predictCandidates.length) {
                    e.preventDefault();
                    acceptPrediction(num);
                    return;
                }
            }
        }

        // Tab -> fetch predictions (when popup not visible)
        if (e.key === 'Tab' && !e.shiftKey && !e.isComposing && promptInput.value.trim()) {
            e.preventDefault();
            if (!predictFetching) {
                fetchPredictions();
            }
            return;
        }

        if (e.key === 'Enter' && e.shiftKey && !e.isComposing) {
            e.preventDefault();
            hidePredictPopup();
            sendPrompt();
        }
    });

    promptInput.addEventListener('input', function () {
        autoResize();
        // Hide prediction popup when user continues typing
        if (predictPopup.style.display !== 'none') {
            hidePredictPopup();
        }
    });

    promptInput.addEventListener('blur', function () {
        // Delay to allow click on popup items
        setTimeout(function () {
            if (!predictPopup.contains(document.activeElement)) {
                hidePredictPopup();
            }
        }, 200);
    });

    function autoResize() {
        promptInput.style.height = 'auto';
        promptInput.style.height = Math.min(promptInput.scrollHeight, 200) + 'px';
    }

    sendBtn.addEventListener('click', sendPrompt);

    queueBtn.addEventListener('click', function () {
        var text = promptInput.value.trim();
        if (text) {
            promptInput.value = '';
            autoResize();
            queue.push({ text: text, auto: false });
            trimQueue();
            queueArea.style.display = 'block';
            renderQueue();
            saveQueue();
            queueArea.scrollTop = queueArea.scrollHeight;
        } else {
            if (queueArea.style.display === 'none' || queueArea.style.display === '') {
                queueArea.style.display = 'block';
                renderQueue();
                queueArea.scrollTop = queueArea.scrollHeight;
            } else {
                queueArea.style.display = 'none';
            }
        }
    });

    cancelBtn.addEventListener('click', cancelRequest);

    document.getElementById('save-chat-btn').addEventListener('click', saveChatAsMarkdown);

    document.getElementById('clear-chat-btn').addEventListener('click', function () {
        chatArea.innerHTML = '';
        chatHistory = [];
        currentAssistantMsg = null;
        currentAssistantText = '';
        needsParagraphBreak = false;
        busy = false;
        cancelBtn.disabled = true;
        stopThinkingTimer();
        queue = [];
        queuePos = 0;
        pendingPrompt = false;
        localStorage.removeItem(HISTORY_KEY);
        // Clear server-side conversation history
        fetch(apiUrl('api/clear'), { method: 'POST' }).catch(function () {});
        promptInput.focus();
    });

    document.getElementById('logout-btn').addEventListener('click', function () {
        // Close SSE connection
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        // Clear server-side history
        fetch(apiUrl('api/clear'), { method: 'POST' }).catch(function () {});
        // Clear local state
        localStorage.removeItem('llm-chat-user');
        currentUser = '';
        // Reload page to show login screen
        window.location.reload();
    });

    modelSelect.addEventListener('change', async function () {
        var selected = modelSelect.value;
        var option = modelSelect.options[modelSelect.selectedIndex];
        var isLocal = option && option.getAttribute('data-type') === 'local';

        // Persist selected model across page reloads
        try {
            localStorage.setItem(MODEL_KEY, selected);
            console.log('[coder-agent] model saved: ' + selected + ' verify=' + localStorage.getItem(MODEL_KEY));
        } catch (e) {
            console.error('[coder-agent] model save FAILED:', e);
        }

        // Local models: no server-side /model command needed (model sent per-request)
        if (isLocal) {
            appendMessage('info', 'Switched to local model: ' + selected);
            return;
        }

        // Claude models: update server-side config
        try {
            var resp = await fetch('api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: '/model ' + selected })
            });
            var events = await resp.json();
            events.forEach(handleEvent);
        } catch (e) {
            appendMessage('error', 'Failed to change model: ' + e.message);
        }
    });

    document.getElementById('refresh-models-btn').addEventListener('click', function () {
        loadModels();
    });

    // --- Load app config (title etc.) ---

    function loadConfig() {
        fetch(apiUrl('api/config'))
            .then(function (resp) { return resp.json(); })
            .then(function (cfg) {
                if (cfg.title) {
                    document.title = cfg.title;
                    var h1 = document.querySelector('header h1');
                    if (h1) h1.textContent = cfg.title;
                }
            })
            .catch(function () {
                // keep default title
            });
    }

    // --- Image handling ---

    function addImageFromFile(file) {
        if (!file || !file.type.startsWith('image/')) return;
        // Limit: max 5 images, max 20MB each
        if (pendingImages.length >= 5) {
            appendMessage('error', 'Maximum 5 images per message');
            return;
        }
        if (file.size > 20 * 1024 * 1024) {
            appendMessage('error', 'Image too large (max 20MB): ' + file.name);
            return;
        }
        var reader = new FileReader();
        reader.onload = function (e) {
            pendingImages.push(e.target.result);
            renderImagePreviews();
        };
        reader.readAsDataURL(file);
    }

    function renderImagePreviews() {
        imagePreviewArea.innerHTML = '';
        pendingImages.forEach(function (dataUrl, idx) {
            var item = document.createElement('div');
            item.className = 'image-preview-item';
            var img = document.createElement('img');
            img.src = dataUrl;
            item.appendChild(img);
            var removeBtn = document.createElement('button');
            removeBtn.className = 'remove-image-btn';
            removeBtn.textContent = '\u00d7';
            removeBtn.addEventListener('click', function () {
                pendingImages.splice(idx, 1);
                renderImagePreviews();
            });
            item.appendChild(removeBtn);
            imagePreviewArea.appendChild(item);
        });
    }

    function clearPendingImages() {
        pendingImages = [];
        imagePreviewArea.innerHTML = '';
    }

    // File input button
    imageBtn.addEventListener('click', function () {
        imageFileInput.click();
    });

    imageFileInput.addEventListener('change', function () {
        var files = imageFileInput.files;
        for (var i = 0; i < files.length; i++) {
            addImageFromFile(files[i]);
        }
        imageFileInput.value = '';
    });

    // Paste image from clipboard
    promptInput.addEventListener('paste', function (e) {
        var items = e.clipboardData && e.clipboardData.items;
        if (!items) return;
        for (var i = 0; i < items.length; i++) {
            if (items[i].type.startsWith('image/')) {
                e.preventDefault();
                addImageFromFile(items[i].getAsFile());
            }
        }
    });

    // Drag and drop images
    inputArea.addEventListener('dragover', function (e) {
        e.preventDefault();
        inputArea.style.outline = '2px dashed var(--accent-color)';
    });
    inputArea.addEventListener('dragleave', function () {
        inputArea.style.outline = '';
    });
    inputArea.addEventListener('drop', function (e) {
        e.preventDefault();
        inputArea.style.outline = '';
        var files = e.dataTransfer && e.dataTransfer.files;
        if (!files) return;
        for (var i = 0; i < files.length; i++) {
            addImageFromFile(files[i]);
        }
    });

    // --- URL fetch ---

    var URL_REGEX = /https?:\/\/[^\s<>"']+/g;

    function detectUrls(text) {
        var matches = text.match(URL_REGEX);
        return matches ? matches.filter(function (v, i, a) { return a.indexOf(v) === i; }) : [];
    }

    function renderFetchedUrls() {
        // No-op: fetched URL indicators removed per user request
    }

    async function fetchUrlContent(url) {
        // Check if already fetched
        for (var i = 0; i < fetchedUrls.length; i++) {
            if (fetchedUrls[i].url === url) return;
        }
        try {
            var resp = await fetch('api/fetch-url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: url })
            });
            var data = await resp.json();
            if (data.ok) {
                fetchedUrls.push({ url: url, content: data.content });
                renderFetchedUrls();
            } else {
                appendMessage('error', 'Fetch failed: ' + (data.error || 'unknown'));
            }
        } catch (e) {
            appendMessage('error', 'Fetch failed: ' + e.message);
        }
    }

    // Show fetch buttons for detected URLs
    var fetchPopupTimeout = null;

    function showFetchButtons() {
        // Remove old fetch button
        var old = document.querySelectorAll('.url-fetch-btn');
        old.forEach(function (el) { el.remove(); });

        var text = promptInput.value;
        var urls = detectUrls(text);
        if (urls.length === 0) return;

        // Filter out already fetched URLs
        var alreadyFetched = fetchedUrls.map(function (f) { return f.url; });
        var newUrls = urls.filter(function (u) { return alreadyFetched.indexOf(u) < 0; });
        if (newUrls.length === 0) return;

        var btn = document.createElement('button');
        btn.className = 'url-fetch-btn';
        btn.textContent = 'Fetch ' + newUrls.length + ' URL' + (newUrls.length > 1 ? 's' : '');
        btn.title = newUrls.join('\n');
        btn.addEventListener('click', function () {
            btn.textContent = 'Fetching...';
            btn.disabled = true;
            Promise.all(newUrls.map(function (url) { return fetchUrlContent(url); }))
                .then(function () {
                    btn.textContent = 'Fetched ' + newUrls.length + ' URL' + (newUrls.length > 1 ? 's' : '');
                });
        });
        imagePreviewArea.appendChild(btn);
    }

    // Debounced URL detection on input
    promptInput.addEventListener('input', function () {
        if (fetchPopupTimeout) clearTimeout(fetchPopupTimeout);
        fetchPopupTimeout = setTimeout(showFetchButtons, 500);
    });

    // Also detect on paste
    promptInput.addEventListener('paste', function () {
        setTimeout(showFetchButtons, 100);
    });

    // --- Init: load config, models, restore history/queue, connect EventSource ---
    loadConfig();
    loadModels();
    restoreHistory();
    restoreQueue();
    connectSSE();
    promptInput.focus();

    } // end doInitApp
})();
