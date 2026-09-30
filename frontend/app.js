/**
 * ================================================================
 * DocuIntel Frontend Controller
 * ================================================================
 *
 * Handles:
 * - Document upload
 * - Document listing
 * - Document deletion
 * - RAG chat
 * - SSE streaming
 * - Markdown rendering
 * - Settings
 * - Sidebar
 * ================================================================
 */

document.addEventListener('DOMContentLoaded', () => {

  // ================================================================
  // STATE
  // ================================================================

  let documents = [];
  let selectedDocId = null;
  let isGenerating = false;


  // ================================================================
  // DOM ELEMENTS
  // ================================================================

  const sidebarEl =
    document.getElementById('sidebar');

  const btnToggleSidebarEl =
    document.getElementById('btnToggleSidebar');

  const btnToggleSidebarMobileEl =
    document.getElementById('btnToggleSidebarMobile');

  const btnNewChatEl =
    document.getElementById('btnNewChat');

  const btnSidebarUploadEl =
    document.getElementById('btnSidebarUpload');

  const documentListEl =
    document.getElementById('documentList');

  const activeModelLabelEl =
    document.getElementById('activeModelLabel');

  const chatViewportEl =
    document.getElementById('chatViewport');

  const chatContainerEl =
    document.getElementById('chatContainer');

  const welcomeContainerEl =
    document.getElementById('welcomeContainer');

  const chatInputEl =
    document.getElementById('chatInput');

  const btnSendEl =
    document.getElementById('btnSend');

  const btnAttachEl =
    document.getElementById('btnAttach');

  const fileInputEl =
    document.getElementById('fileInput');

  const btnResetCorpusEl =
    document.getElementById('btnResetCorpus');


  // Settings

  const settingsModalEl =
    document.getElementById('settingsModal');

  const btnOpenSettingsEl =
    document.getElementById('btnOpenSettings');

  const btnCloseSettingsEl =
    document.getElementById('btnCloseSettings');

  const btnCancelSettingsEl =
    document.getElementById('btnCancelSettings');

  const btnSaveSettingsEl =
    document.getElementById('btnSaveSettings');

  const settingProviderEl =
    document.getElementById('settingProvider');

  const apiKeyGroupEl =
    document.getElementById('apiKeyGroup');

  const settingApiKeyEl =
    document.getElementById('settingApiKey');

  const settingChunkStrategyEl =
    document.getElementById('settingChunkStrategy');


  // ================================================================
  // INITIALIZATION
  // ================================================================

  fetchDocuments();
  fetchSystemStats();


  // ================================================================
  // SIDEBAR
  // ================================================================

  if (btnToggleSidebarEl) {

    btnToggleSidebarEl.addEventListener(
      'click',
      () => {

        sidebarEl.classList.toggle(
          'collapsed'
        );

      }
    );

  }


  if (btnToggleSidebarMobileEl) {

    btnToggleSidebarMobileEl.addEventListener(
      'click',
      () => {

        sidebarEl.classList.toggle(
          'open'
        );

      }
    );

  }


  // ================================================================
  // NEW CHAT
  // ================================================================

  if (btnNewChatEl) {

    btnNewChatEl.addEventListener(
      'click',
      () => {
        startNewChat();
      }
    );

  }


  // ================================================================
  // FILE UPLOAD
  // ================================================================

  if (btnAttachEl) {

    btnAttachEl.addEventListener(
      'click',
      () => {
        fileInputEl.click();
      }
    );

  }


  if (btnSidebarUploadEl) {

    btnSidebarUploadEl.addEventListener(
      'click',
      () => {
        fileInputEl.click();
      }
    );

  }


  if (fileInputEl) {

    fileInputEl.addEventListener(
      'change',
      () => {

        if (
          fileInputEl.files &&
          fileInputEl.files.length > 0
        ) {

          uploadFiles(fileInputEl.files);

        }

      }
    );

  }


  // ================================================================
  // CHAT INPUT
  // ================================================================

  chatInputEl.addEventListener(
    'input',
    () => {

      chatInputEl.style.height = 'auto';

      chatInputEl.style.height =
        Math.min(
          chatInputEl.scrollHeight,
          160
        ) + 'px';

    }
  );


  chatInputEl.addEventListener(
    'keydown',
    (e) => {

      if (
        e.key === 'Enter' &&
        !e.shiftKey
      ) {

        e.preventDefault();

        const query =
          chatInputEl.value.trim();

        if (
          query &&
          !isGenerating
        ) {

          submitQuery(query);

        }

      }

    }
  );


  if (btnSendEl) {

    btnSendEl.addEventListener(
      'click',
      () => {

        const query =
          chatInputEl.value.trim();

        if (
          query &&
          !isGenerating
        ) {

          submitQuery(query);

        }

      }
    );

  }


  // ================================================================
  // RESET SAMPLE CORPUS
  // ================================================================

  if (btnResetCorpusEl) {

    btnResetCorpusEl.addEventListener(
      'click',
      async () => {

        const confirmed =
          confirm(
            'Restore sample documents to initial knowledge base?'
          );

        if (!confirmed) return;

        try {

          const response =
            await fetch(
              '/api/documents/reset-samples',
              {
                method: 'POST'
              }
            );

          if (response.ok) {

            await fetchDocuments();

            startNewChat();

          } else {

            alert(
              'Failed to restore sample documents.'
            );

          }

        } catch (error) {

          alert(
            'Failed to reset: ' +
            error.message
          );

        }

      }
    );

  }


  // ================================================================
  // SETTINGS
  // ================================================================

  if (btnOpenSettingsEl) {

    btnOpenSettingsEl.addEventListener(
      'click',
      () => {

        settingsModalEl.classList.add(
          'active'
        );

      }
    );

  }


  if (btnCloseSettingsEl) {

    btnCloseSettingsEl.addEventListener(
      'click',
      () => {

        settingsModalEl.classList.remove(
          'active'
        );

      }
    );

  }


  if (btnCancelSettingsEl) {

    btnCancelSettingsEl.addEventListener(
      'click',
      () => {

        settingsModalEl.classList.remove(
          'active'
        );

      }
    );

  }


  if (settingProviderEl) {

    settingProviderEl.addEventListener(
      'change',
      () => {

        const value =
          settingProviderEl.value;

        apiKeyGroupEl.style.display =
          (
            value === 'gemini' ||
            value === 'openai'
          )
            ? 'block'
            : 'none';

      }
    );

  }


  if (btnSaveSettingsEl) {

    btnSaveSettingsEl.addEventListener(
      'click',
      async () => {

        const payload = {

          provider:
            settingProviderEl.value,

          api_key:
            settingApiKeyEl.value.trim(),

          chunk_strategy:
            settingChunkStrategyEl.value

        };


        try {

          const response =
            await fetch(
              '/api/system/config',
              {
                method: 'POST',

                headers: {
                  'Content-Type':
                    'application/json'
                },

                body:
                  JSON.stringify(payload)

              }
            );


          if (response.ok) {

            settingsModalEl.classList.remove(
              'active'
            );

            await fetchSystemStats();

            await fetchDocuments();

          } else {

            const errorData =
              await response.json();

            alert(
              'Failed to save settings: ' +
              (
                errorData.detail ||
                'Unknown error'
              )
            );

          }

        } catch (error) {

          alert(
            'Failed to save settings: ' +
            error.message
          );

        }

      }
    );

  }


  // ================================================================
  // FETCH DOCUMENTS
  // ================================================================

  async function fetchDocuments() {

    try {

      const response =
        await fetch(
          '/api/documents'
        );

      if (!response.ok) {
        throw new Error(
          `HTTP ${response.status}`
        );
      }

      const data =
        await response.json();

      documents =
        data.documents || [];

      renderDocumentList(
        documents
      );

    } catch (error) {

      console.error(
        'Error loading documents:',
        error
      );

    }

  }


  // ================================================================
  // FETCH SYSTEM STATS
  // ================================================================

  async function fetchSystemStats() {

    try {

      const response =
        await fetch(
          '/api/system/stats'
        );

      if (!response.ok) {
        throw new Error(
          `HTTP ${response.status}`
        );
      }

      const data =
        await response.json();

      if (activeModelLabelEl) {

        activeModelLabelEl.textContent =
          data.active_model ||
          'Claude Local Engine';

      }

    } catch (error) {

      console.error(
        'Error loading stats:',
        error
      );

    }

  }


  // ================================================================
  // UPLOAD DOCUMENTS
  // ================================================================

  async function uploadFiles(files) {
    if (!files || files.length === 0) return;
    const fileCount = files.length;
    const formData = new FormData();

    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    try {
      const response = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const uploadData = await response.json();
        const uploadedDocs = uploadData.documents || [];
        const actualCount = uploadedDocs.length || fileCount;

        fileInputEl.value = '';
        await fetchDocuments();

        if (uploadedDocs.length === 1) {
          selectedDocId = uploadedDocs[0].doc_id;
          renderDocumentList(documents);
        }

        const docNames = uploadedDocs.length > 0 
          ? uploadedDocs.map(d => `**${escapeHtml(d.title || d.filename)}**`).join(', ')
          : `**${actualCount} document(s)**`;

        appendAssistantMessage(
          `I've successfully processed and indexed ${docNames}. You can now ask questions about them.`
        );
      } else {
        const errorData = await response.json();
        alert('Upload failed: ' + (errorData.detail || 'Error'));
      }
    } catch (error) {
      alert('Upload error: ' + error.message);
    }
  }


  // ================================================================
  // DOCUMENT LIST
  // ================================================================

  function renderDocumentList(docs) {

    documentListEl.innerHTML = '';


    if (
      !docs ||
      docs.length === 0
    ) {

      documentListEl.innerHTML =
        `
        <div
          style="
            color:var(--text-subtle);
            font-size:0.75rem;
            text-align:center;
            padding:12px;
          "
        >
          No documents uploaded.
        </div>
        `;

      return;

    }


    docs.forEach(
      (doc) => {

        const item =
          document.createElement(
            'div'
          );

        item.className =
          `doc-item ${doc.doc_id === selectedDocId
            ? 'active'
            : ''
          }`;


        const title =
          doc.title ||
          doc.filename ||
          'Untitled document';


        item.innerHTML = `

          <span
            class="doc-item-title"
            title="${escapeHtml(title)}"
          >
            ${escapeHtml(title)}
          </span>

          <button
            class="btn-delete-doc"
            title="Remove document"
          >
            ✕
          </button>

        `;


        item.addEventListener(
          'click',
          (event) => {

            if (
              event.target.classList.contains(
                'btn-delete-doc'
              )
            ) {

              event.stopPropagation();

              deleteDocument(
                doc.doc_id
              );

              return;

            }


            if (
              selectedDocId ===
              doc.doc_id
            ) {

              selectedDocId = null;

            } else {

              selectedDocId =
                doc.doc_id;

            }


            renderDocumentList(
              documents
            );

          }
        );


        documentListEl.appendChild(
          item
        );

      }
    );

  }


  // ================================================================
  // DELETE DOCUMENT
  // ================================================================

  async function deleteDocument(
    docId
  ) {

    const confirmed =
      confirm(
        'Remove this document from knowledge base?'
      );

    if (!confirmed) return;


    try {

      const response =
        await fetch(
          `/api/documents/${docId}`,
          {
            method: 'DELETE'
          }
        );


      if (response.ok) {

        if (
          selectedDocId === docId
        ) {

          selectedDocId = null;

        }

        await fetchDocuments();

      } else {

        const errorData =
          await response.json();

        alert(
          'Delete failed: ' +
          (
            errorData.detail ||
            'Unknown error'
          )
        );

      }

    } catch (error) {

      alert(
        'Delete error: ' +
        error.message
      );

    }

  }


  // ================================================================
  // NEW CHAT
  // ================================================================

  function startNewChat() {

    const messages =
      chatContainerEl.querySelectorAll(
        '.message-row'
      );


    messages.forEach(
      message => message.remove()
    );


    if (welcomeContainerEl) {

      welcomeContainerEl.style.display =
        'flex';

    }


    chatInputEl.value = '';

    chatInputEl.style.height =
      'auto';

  }


  // ================================================================
  // SUBMIT RAG QUERY
  // ================================================================

  async function submitQuery(
    queryText
  ) {

    if (isGenerating) return;


    isGenerating = true;

    btnSendEl.disabled = true;

    chatInputEl.value = '';

    chatInputEl.style.height =
      'auto';


    // Hide welcome

    if (welcomeContainerEl) {

      welcomeContainerEl.style.display =
        'none';

    }


    // User message

    appendUserMessage(
      queryText
    );


    // Assistant placeholder

    const {
      bodyEl,
      cursorEl
    } =
      appendAssistantMessagePlaceholder();


    let fullResponse = '';


    try {

      const response =
        await fetch(
          '/api/rag/chat',
          {
            method: 'POST',

            headers: {
              'Content-Type':
                'application/json'
            },

            body:
              JSON.stringify({
                query: queryText,
                top_k: 4,
                doc_filter: selectedDocId || undefined
              })

          }
        );


      if (!response.ok) {

        throw new Error(
          `Server returned HTTP ${response.status}`
        );

      }


      if (!response.body) {

        throw new Error(
          'Streaming response is not available.'
        );

      }


      const reader =
        response.body.getReader();

      const decoder =
        new TextDecoder(
          'utf-8'
        );

      let buffer = '';


      // ============================================================
      // SSE STREAM
      // ============================================================

      while (true) {

        const {
          done,
          value
        } =
          await reader.read();


        if (done) break;


        buffer +=
          decoder.decode(
            value,
            {
              stream: true
            }
          );


        const lines =
          buffer.split('\n');


        buffer =
          lines.pop() || '';


        for (
          const line of lines
        ) {

          const trimmed =
            line.trim();


          if (
            !trimmed.startsWith(
              'data: '
            )
          ) {

            continue;

          }


          const jsonString =
            trimmed.substring(6);


          try {

            const event =
              JSON.parse(
                jsonString
              );


            // ======================================================
            // TOKEN
            // ======================================================

            if (
              event.type ===
              'token'
            ) {

              fullResponse +=
                event.token || '';


              bodyEl.innerHTML =
                formatMarkdown(
                  fullResponse
                );


              scrollToBottom();

            }


            // ======================================================
            // DONE
            // ======================================================

            if (
              event.type ===
              'done'
            ) {

              // Final rendering
              bodyEl.innerHTML =
                formatMarkdown(
                  fullResponse
                );

            }


            // ======================================================
            // ERROR
            // ======================================================

            if (
              event.type ===
              'error'
            ) {

              bodyEl.innerHTML =
                `
                <div class="rag-error">
                  ${escapeHtml(
                  event.message ||
                  'An error occurred.'
                )}
                </div>
                `;

            }

          } catch (error) {

            console.error(
              'Error parsing SSE event:',
              error
            );

          }

        }

      }

    } catch (error) {

      console.error(
        'RAG request error:',
        error
      );


      bodyEl.innerHTML =
        `
        <div class="rag-error">
          Sorry, an error occurred while
          generating the response.
          <br>
          <small>
            ${escapeHtml(
          error.message
        )}
          </small>
        </div>
        `;

    } finally {

      // Remove typing cursor

      if (
        cursorEl &&
        cursorEl.parentNode
      ) {

        cursorEl.parentNode.removeChild(
          cursorEl
        );

      }


      isGenerating = false;

      btnSendEl.disabled = false;

      scrollToBottom();

    }

  }


  // ================================================================
  // USER MESSAGE
  // ================================================================

  function appendUserMessage(
    text
  ) {

    const row =
      document.createElement(
        'div'
      );

    row.className =
      'message-row user';


    const bubble =
      document.createElement(
        'div'
      );

    bubble.className =
      'user-bubble';


    bubble.textContent =
      text;


    row.appendChild(
      bubble
    );


    chatContainerEl.appendChild(
      row
    );


    scrollToBottom();

  }


  // ================================================================
  // ASSISTANT PLACEHOLDER
  // ================================================================

  function appendAssistantMessagePlaceholder() {

    const row =
      document.createElement(
        'div'
      );

    row.className =
      'message-row assistant';


    const avatar =
      document.createElement(
        'div'
      );

    avatar.className =
      'assistant-avatar';


    avatar.innerHTML = `

      <svg
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
      >

        <path
          d="
            M12 2L2 7l10 5
            10-5-10-5z
            M2 17l10 5 10-5
            M2 12l10 5 10-5
          "
        />

      </svg>

    `;


    const body =
      document.createElement(
        'div'
      );

    body.className =
      'assistant-body';


    const cursor =
      document.createElement(
        'span'
      );

    cursor.className =
      'typing-cursor';


    body.appendChild(
      cursor
    );


    row.appendChild(
      avatar
    );

    row.appendChild(
      body
    );


    chatContainerEl.appendChild(
      row
    );


    scrollToBottom();


    return {
      bodyEl: body,
      cursorEl: cursor
    };

  }


  // ================================================================
  // STATIC ASSISTANT MESSAGE
  // ================================================================

  function appendAssistantMessage(
    text
  ) {

    if (welcomeContainerEl) {

      welcomeContainerEl.style.display =
        'none';

    }


    const row =
      document.createElement(
        'div'
      );

    row.className =
      'message-row assistant';


    const avatar =
      document.createElement(
        'div'
      );

    avatar.className =
      'assistant-avatar';


    avatar.innerHTML = `

      <svg
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
      >

        <path
          d="
            M12 2L2 7l10 5
            10-5-10-5z
            M2 17l10 5 10-5
            M2 12l10 5 10-5
          "
        />

      </svg>

    `;


    const body =
      document.createElement(
        'div'
      );

    body.className =
      'assistant-body';


    body.innerHTML =
      formatMarkdown(text);


    row.appendChild(
      avatar
    );

    row.appendChild(
      body
    );


    chatContainerEl.appendChild(
      row
    );


    scrollToBottom();

  }


  // ================================================================
  // MARKDOWN RENDERER
  // ================================================================

  function formatMarkdown(
    text
  ) {

    if (!text) {
      return '';
    }


    // Make sure marked exists

    if (
      typeof marked ===
      'undefined'
    ) {

      console.error(
        'Marked.js is not loaded.'
      );

      return escapeHtml(
        text
      );

    }


    // Configure Markdown

    marked.setOptions({

      gfm: true,

      breaks: true,

      headerIds: false,

      mangle: false

    });


    // Convert Markdown -> HTML

    const html =
      marked.parse(
        text
      );


    // Sanitize generated HTML

    if (
      typeof DOMPurify !==
      'undefined'
    ) {

      return DOMPurify.sanitize(
        html,
        {
          USE_PROFILES: {
            html: true
          }
        }
      );

    }


    return html;

  }


  // ================================================================
  // SCROLL
  // ================================================================

  function scrollToBottom() {

    if (!chatViewportEl) {
      return;
    }


    chatViewportEl.scrollTop =
      chatViewportEl.scrollHeight;

  }


  // ================================================================
  // HTML ESCAPE
  // ================================================================

  function escapeHtml(
    value
  ) {

    if (
      value === null ||
      value === undefined
    ) {

      return '';

    }


    return String(value)

      .replace(
        /&/g,
        '&amp;'
      )

      .replace(
        /</g,
        '&lt;'
      )

      .replace(
        />/g,
        '&gt;'
      )

      .replace(
        /"/g,
        '&quot;'
      )

      .replace(
        /'/g,
        '&#039;'
      );

  }

});