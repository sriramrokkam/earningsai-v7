// Constants
const MAX_FILE_SIZE = 20 * 1024 * 1024;
const ALLOWED_TYPES = [
    'application/pdf',
    'text/plain',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel',
    'image/jpeg',
    'image/png'
];

// Global Variables
let recognition = null;
let isRecording = false;
let uploadStatusInterval = null;

// Speech Recognition
function initializeSpeechRecognition() {
    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onstart = () => {
            isRecording = true;
            updateMicButtonState();
        };
        
        recognition.onend = () => {
            isRecording = false;
            updateMicButtonState();
        };
        
        recognition.onresult = (event) => {
            document.getElementById('userInput').value = event.results[0][0].transcript;
        };
        
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            isRecording = false;
            updateMicButtonState();
        };
    } else {
        console.error('Speech recognition not supported');
        document.getElementById('micButton').style.display = 'none';
    }
}

function updateMicButtonState() {
    const micButton = document.getElementById('micButton');
    const recordingIndicator = document.getElementById('recordingIndicator');
    
    micButton.classList.toggle('recording', isRecording);
    recordingIndicator.classList.toggle('active', isRecording);
}

function toggleRecording() {
    if (!recognition) initializeSpeechRecognition();
    isRecording ? recognition.stop() : recognition.start();
}

// Utility Functions
function copyResponse() {
    const responseContent = document.querySelector('.response-content-wrapper');
    if (!responseContent) return;

    navigator.clipboard.writeText(responseContent.innerText)
        .then(() => alert('Response copied to clipboard!'))
        .catch((error) => {
            console.error('Failed to copy:', error);
            alert('Failed to copy response.');
        });
}

function exportToPDF() {
    if (!window.jspdf || !window.html2canvas) {
        alert('PDF export libraries failed to load. Please refresh the page.');
        return;
    }

    const { jsPDF } = window.jspdf;
    const responseCard = document.getElementById('responseCard');
    const userInput = document.getElementById('userInput').value.trim();

    if (!responseCard) {
        alert('No response available to export.');
        return;
    }

    const exportButton = document.getElementById('exportButton');
    const originalInnerHTML = exportButton.innerHTML;
    exportButton.innerHTML = '<div class="spinner"></div>';
    exportButton.disabled = true;

    // Create a container for combined content
    const combinedContent = document.createElement('div');
    combinedContent.style.width = '210mm'; // A4 width
    combinedContent.style.padding = '10mm';
    combinedContent.style.background = '#fff';
    combinedContent.style.fontFamily = 'Arial, sans-serif';
    combinedContent.style.position = 'absolute';
    combinedContent.style.left = '-9999px'; // Off-screen
    document.body.appendChild(combinedContent);

    // Clone the response card’s styles for consistency
    const responseStyles = window.getComputedStyle(responseCard);
    combinedContent.style.fontSize = responseStyles.fontSize;
    combinedContent.style.color = responseStyles.color;

    // Add user input section with similar styling to response card
    if (userInput) {
        const inputSection = document.createElement('div');
        inputSection.className = 'card response-card'; // Reuse response-card class
        inputSection.innerHTML = `
            <div class="response-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h1 class="response-title" style="font-size: 1.5em; margin: 0;">USER INPUT</h1>
            </div>
            <div class="response-content-wrapper" style="padding: 10px;">
                <div class="response-content" style="font-size: 1em;">
                    ${formatFinancialResponse(userInput)} <!-- Apply the same formatting -->
                </div>
            </div>
        `;
        combinedContent.appendChild(inputSection);
    }

    // Clone and append the response card
    const responseClone = responseCard.cloneNode(true);
    combinedContent.appendChild(responseClone);

    // Ensure rendering before capturing
    setTimeout(() => {
        html2canvas(combinedContent, {
            scale: 2,
            useCORS: true,
            logging: true,
            windowWidth: 794, // A4 width in pixels at 96 DPI
            windowHeight: 1123 // A4 height in pixels at 96 DPI
        })
            .then((canvas) => {
                const imgData = canvas.toDataURL('image/png');
                const pdf = new jsPDF('p', 'mm', 'a4');
                const imgWidth = 210; // A4 width in mm
                const imgHeight = (canvas.height * imgWidth) / canvas.width;
                let heightLeft = imgHeight;
                let position = 0;

                pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
                heightLeft -= 297; // A4 height in mm

                while (heightLeft > 0) {
                    position -= 297;
                    pdf.addPage();
                    pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
                    heightLeft -= 297;
                }

                pdf.save('Finsight_Earnings_Response.pdf');
            })
            .catch((error) => {
                console.error('PDF export failed:', error);
                alert('Failed to export to PDF. Check console for details.');
            })
            .finally(() => {
                exportButton.innerHTML = originalInnerHTML;
                exportButton.disabled = false;
                document.body.removeChild(combinedContent); // Clean up
            });
    }, 100); // Delay for rendering
}

// Response Formatting Functions
function formatFinancialResponse(rawResponse) {
    // Check if the response contains HTML
    if (/<[a-z][\s\S]*>/i.test(rawResponse)) {
        return handleHtmlResponse(rawResponse);
    } else {
        return handlePlainTextResponse(rawResponse);
    }
}

function handleHtmlResponse(rawResponse) {
    // Basic sanitization and table extraction (you might want a proper sanitizer like DOMPurify in production)
    const parser = new DOMParser();
    const doc = parser.parseFromString(`<div>${rawResponse}</div>`, 'text/html');
    const tables = doc.getElementsByTagName('table');
    let html = '';

    // Process each node in the response
    Array.from(doc.body.firstChild.childNodes).forEach(node => {
        if (node.nodeName === 'TABLE') {
            html += formatTable(node.outerHTML);
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            // Preserve other HTML elements
            html += node.outerHTML;
        } else if (node.nodeType === Node.TEXT_NODE && node.textContent.trim()) {
            html += `<p>${formatFinancialItem(node.textContent.trim())}</p>`;
        }
    });

    return html;
}

function formatTable(tableHtml) {
    // Simply return the table HTML to be styled by CSS
    // Add any additional reformatting here if needed (e.g., adding classes)
    return tableHtml;
}

function handlePlainTextResponse(rawResponse) {
    const lines = rawResponse.split('\n').filter(line => line.trim());
    let html = '';
    let currentSection = [];
    let inList = false;

    lines.forEach((line, index) => {
        const trimmedLine = line.trim();

        if (isHeading(trimmedLine) && currentSection.length) {
            html += formatSection(currentSection);
            currentSection = [trimmedLine];
        } else if (trimmedLine.startsWith('•') || trimmedLine.startsWith('-') || trimmedLine.startsWith('*')) {
            if (!inList && currentSection.length) {
                html += formatSection(currentSection);
                currentSection = [];
            }
            inList = true;
            currentSection.push(trimmedLine);
        } else if (trimmedLine.startsWith('Source:') || trimmedLine.startsWith('Driver:')) {
            if (inList && currentSection.length) {
                html += formatList(currentSection);
                inList = false;
            } else if (currentSection.length) {
                html += formatSection(currentSection);
            }
            currentSection = [trimmedLine];
            html += formatSection(currentSection);
            currentSection = [];
        } else {
            if (inList && currentSection.length) {
                html += formatList(currentSection);
                inList = false;
            }
            currentSection.push(trimmedLine);
        }

        if (index === lines.length - 1 && currentSection.length) {
            if (inList) {
                html += formatList(currentSection);
            } else {
                html += formatSection(currentSection);
            }
        }
    });

    return html;
}

function isHeading(line) {
    return /^\d+\.\s/.test(line) || 
           /\w+\s\(\d+%\)/.test(line) || 
           (line.length > 0 && line.length < 50 && !line.includes(':') && !line.startsWith('•') && !line.startsWith('-') && !line.startsWith('*'));
}

function formatSection(lines) {
    if (!lines.length) return '';
    
    let html = '<div>';
    const firstLine = lines[0];

    if (isHeading(firstLine)) {
        html += `<h2>${firstLine}</h2>`;
        lines = lines.slice(1);
    }

    lines.forEach(line => {
        if (line.startsWith('Source:') || line.startsWith('Driver:')) {
            const [key, value] = line.split(':');
            html += `<p class="${key.toLowerCase()}"><strong>${key}:</strong> ${value.trim()}</p>`;
        } else {
            html += `<p>${formatFinancialItem(line)}</p>`;
        }
    });

    html += '</div>';
    return html;
}

function formatList(lines) {
    if (!lines.length) return '';
    
    let html = '<ul>';
    lines.forEach(line => {
        const content = line.replace(/^[-•*]\s*/, '').trim();
        html += `<li>${formatFinancialItem(content)}</li>`;
    });
    html += '</ul>';
    return html;
}

function formatFinancialItem(item) {
    return item.replace(/(\$?\d+\.?\d*[BMT]?)/g, '<span class="financial-number">$1</span>')
              .replace(/(\d+%)/g, '<span class="financial-percentage">$1</span>');
}

// Core Functions
function sendMessage() {
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const responseArea = document.getElementById('responseArea');
    const userInputValue = userInput.value.trim();

    if (!userInputValue) return;

    // Temporarily disable input while processing
    userInput.disabled = true;
    sendButton.disabled = true;
    responseArea.innerHTML = '<div class="card loading">Processing your request. Please standby...</div>';

    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userInputValue })
    })
        .then(response => response.json())
        .then(data => {
            const formattedContent = formatFinancialResponse(data.FINAL_RESULT);
            responseArea.innerHTML = `
                <div class="card response-card" id="responseCard">
                    <div class="response-header">
                        <h1 class="response-title">SUMMARY</h1>
                        <div class="button-group">
                            <button class="btn btn-copy" id="copyButton" title="Copy to Clipboard">
                                <svg class="copy-icon" viewBox="0 0 24 24">
                                    <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
                                </svg>
                            </button>
                            <button class="btn btn-export" id="exportButton" title="Export to PDF">
                                <svg class="export-icon" viewBox="0 0 24 24">
                                    <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z" fill="currentColor"/>
                                </svg>
                            </button>
                        </div>
                    </div>
                    <div class="response-content-wrapper">
                        <div class="response-content">
                            ${formattedContent}
                        </div>
                    </div>
                </div>`;

            document.getElementById('copyButton').addEventListener('click', copyResponse);
            document.getElementById('exportButton').addEventListener('click', exportToPDF);
        })
        .catch(error => {
            console.error('Error:', error);
            responseArea.innerHTML = '<div class="card response-card"><p>Error: Could not process your request.</p></div>';
        })
        .finally(() => {
            // Re-enable input with original value preserved
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.value = userInputValue; // Keep the original input
            userInput.focus();
        });
}
function generateEmbeddings() {
    const generateButton = document.getElementById('generateEmbeddingsButton');
    generateButton.disabled = true;
    generateButton.innerHTML = '<div class="spinner"></div> Generating...';

    fetch('/generate-embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                updateUploadStatus(data.error, 'error');
            } else {
                updateUploadStatus(data.message, 'success');
            }
        })
        .catch(error => {
            console.error('Error generating embeddings:', error);
            updateUploadStatus('Failed to generate embeddings', 'error');
        })
        .finally(() => {
            generateButton.innerHTML = 'Generate Embeddings';
        });
}

// UI Updates
function updateUploadStatus(message, type) {
    const uploadStatus = document.getElementById('uploadStatus');
    uploadStatus.textContent = message;
    uploadStatus.className = `upload-status-${type}`;
    
    if (type !== 'loading') {
        setTimeout(() => {
            uploadStatus.textContent = '';
            uploadStatus.className = '';
        }, 5000);
    }
}

function startUploadAnimation() {
    let dots = 0;
    updateUploadStatus('Uploading', 'loading');
    uploadStatusInterval = setInterval(() => {
        dots = (dots + 1) % 4;
        document.getElementById('uploadStatus').textContent = 'Uploading' + '.'.repeat(dots);
    }, 500);
}

function stopUploadAnimation() {
    if (uploadStatusInterval) {
        clearInterval(uploadStatusInterval);
        uploadStatusInterval = null;
    }
}

function showOverwriteModal(message, onConfirm, onCancel) {
    const modal = document.getElementById('overwriteModal');
    const modalMessage = document.getElementById('modalMessage');
    const confirmBtn = document.getElementById('modalConfirm');
    const cancelBtn = document.getElementById('modalCancel');

    modalMessage.textContent = message;
    modal.style.display = 'flex';

    confirmBtn.onclick = () => {
        modal.style.display = 'none';
        onConfirm();
    };
    cancelBtn.onclick = () => {
        modal.style.display = 'none';
        onCancel();
    };
}

// Event Listeners
function setupEventListeners() {
    const elements = {
        uploadButton: document.getElementById('uploadButton'),
        fileInput: document.getElementById('fileInput'),
        micButton: document.getElementById('micButton'),
        sendButton: document.getElementById('sendButton'),
        generateEmbeddingsButton: document.getElementById('generateEmbeddingsButton'),
        userInput: document.getElementById('userInput')
    };

    elements.uploadButton.addEventListener('click', () => elements.fileInput.click());

    elements.fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;

        if (!ALLOWED_TYPES.includes(file.type)) {
            updateUploadStatus('Unsupported file type. Only PDF, Images, and DOCX allowed.', 'error');
            return;
        }

        if (file.size > MAX_FILE_SIZE) {
            updateUploadStatus(`File too large. Maximum size is ${MAX_FILE_SIZE / 1024 / 1024}MB.`, 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        function uploadFile(overwrite = false) {
            if (overwrite) formData.append('overwrite', 'true');
            
            startUploadAnimation();
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    stopUploadAnimation();
                    if (data.error) {
                        updateUploadStatus(data.error, 'error');
                    } else if (data.exists) {
                        showOverwriteModal(data.message,
                            () => uploadFile(true),
                            () => updateUploadStatus('File already exists', 'error')
                        );
                    } else {
                        updateUploadStatus(data.message, 'success');
                        document.getElementById('generateEmbeddingsButton').disabled = false;
                    }
                })
                .catch(error => {
                    stopUploadAnimation();
                    console.error('Upload error:', error);
                    updateUploadStatus('Upload failed: Network error', 'error');
                })
                .finally(() => {
                    elements.fileInput.value = '';
                });
        }

        uploadFile();
    });

    elements.micButton.addEventListener('click', toggleRecording);
    elements.sendButton.addEventListener('click', sendMessage);
    elements.generateEmbeddingsButton.addEventListener('click', generateEmbeddings);
    
    elements.userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    const generateButton = document.getElementById('generateEmbeddingsButton');
    generateButton.disabled = true;
    
    initializeSpeechRecognition();
    setupEventListeners();
});