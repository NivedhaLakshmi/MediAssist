import { Streamlit } from "streamlit-component-lib";


// ============================================================
// STATE
// ============================================================

let selectedFiles = [];

let mediaRecorder = null;

let audioChunks = [];

let isRecording = false;

let isTranscribing = false;

let lastPythonValue;

let lastVoiceCompletionId;

let cameraStream = null;

let lastFrameHeight = 0;

let isAwaitingResponse = false;

let awaitingTimeout = null;


// ============================================================
// COMPONENT EVENT ID
// ============================================================

function createComponentEventId() {

    return (
        Date.now().toString() +
        "-" +
        Math.random().toString(36).slice(2)
    );

}


// ============================================================
// READ FILE
// ============================================================

function readFile(file) {

    return new Promise((resolve, reject) => {

        const reader =
            new FileReader();


        reader.onload =
            function (event) {

                const base64 =
                    event.target.result
                        .split(",")[1];


                resolve({

                    name: file.name,

                    type: file.type,

                    size: file.size,

                    data: base64

                });

            };


        reader.onerror = reject;


        reader.readAsDataURL(file);

    });

}


// ============================================================
// UPDATE FILE DISPLAY
// ============================================================

function updateFileDisplay() {

    const display =
        document.getElementById(
            "file-display"
        );


    if (!display) {
        return;
    }


    display.innerHTML = "";


    selectedFiles.forEach(
        function (file) {

            const item =
                document.createElement(
                    "div"
                );


            item.className =
                "file-item";


            let icon = "📄";


            if (
                file.type &&
                file.type.startsWith("image/")
            ) {

                icon = "🖼️";

            }


            item.textContent =
                icon + " " + file.name;


            display.appendChild(
                item
            );

        }
    );

}


// ============================================================
// AUTO-GROW TEXTAREA
// ============================================================

const TEXTAREA_MAX_HEIGHT = 180;

function autoGrowTextarea() {

    const messageInput =
        document.getElementById(
            "message"
        );

    if (!messageInput) {
        return;
    }

    messageInput.style.height = "auto";

    messageInput.style.height =
        Math.min(
            messageInput.scrollHeight,
            TEXTAREA_MAX_HEIGHT
        ) + "px";

}


// ============================================================
// SEND FEEDBACK
//
// Purely cosmetic confirmation that the question left the box.
// It is cleared on a short timer rather than on a render event:
// Streamlit only dispatches RENDER_EVENT when the component's
// props change, and this component's props are identical on every
// run, so a render event is not guaranteed to arrive.
// ============================================================

const AWAITING_TIMEOUT_MS = 2500;


function setAwaitingResponse(flag) {

    isAwaitingResponse = flag;

    const messageInput =
        document.getElementById(
            "message"
        );

    // Feedback only. The input is never disabled, so a stuck flag
    // can never lock the user out.
    if (messageInput) {

        messageInput.placeholder = flag
            ? "Sent - generating the answer..."
            : "Type your question...";

    }

    if (awaitingTimeout) {

        clearTimeout(awaitingTimeout);

        awaitingTimeout = null;

    }

    if (flag) {

        awaitingTimeout = setTimeout(
            function () {
                setAwaitingResponse(false);
            },
            AWAITING_TIMEOUT_MS
        );

    }

}


// ============================================================
// UPDATE FRAME HEIGHT
// ============================================================

function updateFrameHeight() {

    const menu =
        document.getElementById(
            "attachment-menu"
        );

    const menuOpen =
        menu &&
        menu.classList.contains("visible");

    const cameraDialog =
        document.getElementById(
            "camera-dialog"
        );

    const cameraOpen =
        cameraDialog &&
        cameraDialog.classList.contains("visible");

    const messageInput =
        document.getElementById(
            "message"
        );

    // Base height tracks the textarea so long questions stay
    // fully visible and the send button never gets clipped.

    let height = 112;

    if (messageInput) {

        const textareaHeight =
            Math.min(
                messageInput.scrollHeight,
                TEXTAREA_MAX_HEIGHT
            );

        height = Math.max(
            112,
            textareaHeight + 66
        );

    }

    if (selectedFiles.length > 0) {
        height += 40;
    }

    if (menuOpen) {
        height = Math.max(
            height,
            270
        );
    }

    if (cameraOpen) {
        height = Math.max(
            height,
            500
        );
    }

    // Only tell Streamlit when the height actually changes.
    // Posting a resize on every keystroke floods the websocket,
    // especially while the server is busy generating an answer.
    if (height !== lastFrameHeight) {

        lastFrameHeight = height;

        Streamlit.setFrameHeight(height);

    }
}


// ============================================================
// CLOSE ATTACHMENT MENU
// ============================================================

function closeAttachmentMenu() {

    const menu =
        document.getElementById(
            "attachment-menu"
        );


    if (!menu) {
        return;
    }


    menu.classList.remove(
        "visible"
    );


    updateFrameHeight();

}


// ============================================================
// TOGGLE ATTACHMENT MENU
// ============================================================

function toggleAttachmentMenu() {

    const menu =
        document.getElementById(
            "attachment-menu"
        );


    if (!menu) {
        return;
    }


    menu.classList.toggle(
        "visible"
    );


    updateFrameHeight();

}


// ============================================================
// OPEN CAMERA
// ============================================================

function openCamera() {

    const cameraInput =
        document.getElementById(
            "cameraInput"
        );


    closeAttachmentMenu();


    if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
    ) {

        if (cameraInput) {
            cameraInput.click();
        }

        return;

    }


    navigator.mediaDevices.getUserMedia({
        video: {
            facingMode: {
                ideal: "environment"
            }
        },
        audio: false
    })
        .then(
            function (stream) {

                const dialog =
                    document.getElementById(
                        "camera-dialog"
                    );

                const video =
                    document.getElementById(
                        "camera-preview"
                    );

                if (!dialog || !video) {

                    stream.getTracks().forEach(
                        function (track) {
                            track.stop();
                        }
                    );

                    return;

                }


                cameraStream = stream;

                video.srcObject = stream;

                dialog.classList.add(
                    "visible"
                );

                updateFrameHeight();

            }
        )
        .catch(
            function (error) {

                console.error(
                    "Camera error:",
                    error
                );


                // Some browsers block camera access in an
                // embedded frame. Let mobile browsers use their
                // native camera picker in that case.
                if (cameraInput) {
                    cameraInput.click();
                }

            }
        );

}


// ============================================================
// OPEN IMAGE UPLOAD
// ============================================================

function openImageUpload() {

    const imageInput =
        document.getElementById(
            "imageInput"
        );


    if (!imageInput) {
        return;
    }


    closeAttachmentMenu();


    imageInput.click();

}


// ============================================================
// OPEN PDF UPLOAD
// ============================================================

function openPdfUpload() {

    const pdfInput =
        document.getElementById(
            "pdfInput"
        );


    if (!pdfInput) {
        return;
    }


    closeAttachmentMenu();


    pdfInput.click();

}


// ============================================================
// HANDLE SELECTED FILES
// ============================================================

function addSelectedFiles(files) {


    if (
        files.length === 0
    ) {

        return;

    }


    const supportedFiles =
        files.filter(
            function (file) {

                const isPDF =
                    file.type ===
                        "application/pdf"
                    ||
                    file.name
                        .toLowerCase()
                        .endsWith(".pdf");


                const isImage =
                    file.type &&
                    file.type.startsWith(
                        "image/"
                    );


                return (
                    isPDF ||
                    isImage
                );

            }
        );


    selectedFiles =
        selectedFiles.concat(
            supportedFiles
        );


    updateFileDisplay();


    updateFrameHeight();

}


function handleSelectedFiles(
    event
) {

    const files =
        Array.from(
            event.target.files || []
        );


    addSelectedFiles(files);


    // Allow the same file to be
    // selected again.

    if (event.target) {
        event.target.value = "";
    }

}


// ============================================================
// CAMERA CAPTURE
// ============================================================

function closeCamera() {

    const dialog =
        document.getElementById(
            "camera-dialog"
        );

    const video =
        document.getElementById(
            "camera-preview"
        );

    if (cameraStream) {

        cameraStream.getTracks().forEach(
            function (track) {
                track.stop();
            }
        );

        cameraStream = null;

    }

    if (video) {
        video.srcObject = null;
    }

    if (dialog) {
        dialog.classList.remove(
            "visible"
        );
    }

    updateFrameHeight();

}


function captureCameraPhoto() {

    const video =
        document.getElementById(
            "camera-preview"
        );

    if (!video || !video.videoWidth) {
        return;
    }


    const canvas =
        document.createElement(
            "canvas"
        );

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    context.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );

    canvas.toBlob(
        function (blob) {

            if (!blob) {
                return;
            }


            const image =
                new File(
                    [blob],
                    "camera-photo-" + Date.now() + ".jpg",
                    {
                        type: "image/jpeg"
                    }
                );

            addSelectedFiles([image]);

            closeCamera();

        },
        "image/jpeg",
        0.92
    );

}


// ============================================================
// RECEIVE VALUE FROM PYTHON
//
// Used for Whisper transcription.
// ============================================================

function updateTextFromPython(args) {

    if (!args) {
        return;
    }


    const messageInput =
        document.getElementById(
            "message"
        );


    if (!messageInput) {
        return;
    }


    if (
        args.value === undefined ||
        args.value === null ||
        args.value === lastPythonValue
    ) {

        return;

    }


    lastPythonValue =
        args.value;


    if (
        args.value !==
        messageInput.value
    ) {

        messageInput.value =
            args.value;


        messageInput.focus();


        messageInput.selectionStart =
            messageInput.value.length;


        messageInput.selectionEnd =
            messageInput.value.length;


        autoGrowTextarea();

        updateFrameHeight();

    }

}


// ============================================================
// SUBMIT MESSAGE
// ============================================================

let isSubmitting = false;


async function submitMessage() {

    if (isSubmitting) {
        return;
    }

    const messageInput =
        document.getElementById(
            "message"
        );


    if (!messageInput) {
        return;
    }


    const message =
        messageInput.value.trim();


    // Don't submit empty input
    // unless files are attached.

    if (
        !message &&
        selectedFiles.length === 0
    ) {

        return;

    }


    const sendButton =
        document.getElementById(
            "send"
        );


    isSubmitting = true;


    if (sendButton) {

        sendButton.disabled =
            true;

    }


    try {

        const fileData = [];


        // ---------------------------------------------
        // Convert selected files
        //
        // Works for both PDFs and images.
        // ---------------------------------------------

        for (
            const file of selectedFiles
        ) {

            const data =
                await readFile(file);


            fileData.push(
                data
            );

        }


        // ---------------------------------------------
        // Send message to Python
        // ---------------------------------------------

        const eventId =
            createComponentEventId();


        console.log(
            "[chat_input] sending message",
            eventId
        );


        Streamlit.setComponentValue({

            id:
                eventId,

            type:
                "message",

            text:
                message,

            files:
                fileData

        });


        // ---------------------------------------------
        // Clear input
        // ---------------------------------------------

        messageInput.value =
            "";


        selectedFiles = [];


        updateFileDisplay();


        autoGrowTextarea();


        updateFrameHeight();


        setAwaitingResponse(true);

    }


    catch (error) {

        console.error(
            "Error submitting message:",
            error
        );

    }


    finally {

        isSubmitting = false;

        if (sendButton) {

            sendButton.disabled =
                false;

        }

    }

}


// ============================================================
// START RECORDING
// ============================================================

async function startRecording() {

    try {

        const stream =
            await navigator
                .mediaDevices
                .getUserMedia({
                    audio: true
                });


        audioChunks = [];


        mediaRecorder =
            new MediaRecorder(
                stream
            );


        mediaRecorder.ondataavailable =
            function (event) {

                if (
                    event.data &&
                    event.data.size > 0
                ) {

                    audioChunks.push(
                        event.data
                    );

                }

            };


        mediaRecorder.onstop =
            async function () {

                const audioBlob =
                    new Blob(
                        audioChunks,
                        {
                            type:
                                "audio/webm"
                        }
                    );


                console.log(
                    "Audio recorded:",
                    audioBlob.size,
                    "bytes"
                );


                const reader =
                    new FileReader();


                reader.onloadend =
                    function () {

                        const base64 =
                            reader.result
                                .split(",")[1];


                        console.log(
                            "Sending audio to Python"
                        );


                        Streamlit
                            .setComponentValue({

                                id:
                                    createComponentEventId(),

                                type:
                                    "audio",

                                audio:
                                    base64,

                                audio_type:
                                    "audio/webm"

                            });

                    };


                reader.readAsDataURL(
                    audioBlob
                );


                // Stop microphone tracks

                stream
                    .getTracks()
                    .forEach(
                        function (track) {

                            track.stop();

                        }
                    );

            };


        mediaRecorder.start();


        isRecording = true;


        updateMicrophoneButton();

    }


    catch (error) {

        console.error(
            "Microphone error:",
            error
        );


        alert(
            "Unable to access the microphone. " +
            "Please allow microphone access."
        );

    }

}


// ============================================================
// STOP RECORDING
// ============================================================

function stopRecording() {

    if (
        mediaRecorder &&
        mediaRecorder.state !==
            "inactive"
    ) {

        mediaRecorder.stop();

    }


    isRecording = false;

    isTranscribing = true;

    updateMicrophoneButton();

}


// ============================================================
// MICROPHONE TOGGLE
// ============================================================

function toggleMicrophone() {

    if (isTranscribing) {

        return;

    }


    if (isRecording) {

        stopRecording();

    }


    else {

        startRecording();

    }

}


// ============================================================
// MICROPHONE BUTTON UI
// ============================================================

// White line icons so the control matches the dark input bar.

const MIC_ICON_IDLE =
    '<svg class="control-icon" viewBox="0 0 24 24" aria-hidden="true">' +
    '<rect x="9" y="3" width="6" height="11" rx="3"/>' +
    '<path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>';

const MIC_ICON_STOP =
    '<svg class="control-icon" viewBox="0 0 24 24" aria-hidden="true">' +
    '<rect x="6" y="6" width="12" height="12" rx="2"/></svg>';

const MIC_ICON_BUSY =
    '<svg class="control-icon" viewBox="0 0 24 24" aria-hidden="true">' +
    '<path d="M12 3a9 9 0 1 0 9 9"/></svg>';

function updateMicrophoneButton() {

    const microphone =
        document.getElementById(
            "microphone"
        );


    if (!microphone) {
        return;
    }


    if (isTranscribing) {

        microphone.innerHTML = MIC_ICON_BUSY;

        microphone.title = "Transcribing...";

        microphone.disabled = true;

        microphone.classList.remove("recording");

    }


    else if (isRecording) {

        microphone.innerHTML = MIC_ICON_STOP;

        microphone.title = "Stop recording";

        microphone.disabled = false;

        microphone.classList.add("recording");

    }


    else {

        microphone.innerHTML = MIC_ICON_IDLE;

        microphone.title = "Voice input";

        microphone.disabled = false;

        microphone.classList.remove("recording");

    }

}


// ============================================================
// KEYBOARD
// ============================================================

function handleKeyDown(event) {

    if (
        event.key === "Enter" &&
        !event.shiftKey
    ) {

        event.preventDefault();

        event.stopPropagation();

        submitMessage();

    }

}


// ============================================================
// COMPONENT RENDER
// ============================================================

function onRender(event) {

    console.log("[chat_input] render event received");

    setAwaitingResponse(false);

    const args =
        event.detail &&
        event.detail.args;


    // ---------------------------------------------
    // Voice transcription completion
    // ---------------------------------------------

    if (
        args &&
        args.voice_completion_id !==
            lastVoiceCompletionId
    ) {

        lastVoiceCompletionId =
            args.voice_completion_id;


        isTranscribing =
            false;

    }


    // ---------------------------------------------
    // Find controls
    // ---------------------------------------------

    const sendButton =
        document.getElementById(
            "send"
        );


    const messageInput =
        document.getElementById(
            "message"
        );


    const attachButton =
        document.getElementById(
            "attach"
        );


    const cameraInput =
        document.getElementById(
            "cameraInput"
        );


    const imageInput =
        document.getElementById(
            "imageInput"
        );


    const pdfInput =
        document.getElementById(
            "pdfInput"
        );


    const cameraOption =
        document.getElementById(
            "camera-option"
        );


    const imageOption =
        document.getElementById(
            "image-option"
        );


    const pdfOption =
        document.getElementById(
            "pdf-option"
        );


    const closeCameraButton =
        document.getElementById(
            "close-camera"
        );


    const captureCameraButton =
        document.getElementById(
            "capture-camera"
        );


    const microphone =
        document.getElementById(
            "microphone"
        );


    if (
        !sendButton ||
        !messageInput ||
        !attachButton ||
        !cameraInput ||
        !imageInput ||
        !pdfInput ||
        !cameraOption ||
        !imageOption ||
        !pdfOption ||
        !closeCameraButton ||
        !captureCameraButton ||
        !microphone
    ) {

        return;

    }


    // ---------------------------------------------
    // Receive transcription from Python
    // ---------------------------------------------

    updateTextFromPython(
        args
    );


    // ---------------------------------------------
    // Event handlers
    //
    // Assigned on every render because Streamlit
    // may re-render the component.
    // ---------------------------------------------

    sendButton.onclick =
        submitMessage;


    attachButton.onclick =
        toggleAttachmentMenu;


    cameraOption.onclick =
        openCamera;


    imageOption.onclick =
        openImageUpload;


    pdfOption.onclick =
        openPdfUpload;


    cameraInput.onchange =
        handleSelectedFiles;


    imageInput.onchange =
        handleSelectedFiles;


    pdfInput.onchange =
        handleSelectedFiles;


    closeCameraButton.onclick =
        closeCamera;


    captureCameraButton.onclick =
        captureCameraPhoto;


    messageInput.onkeydown =
        handleKeyDown;


    messageInput.oninput =
        function () {

            autoGrowTextarea();

            updateFrameHeight();

        };


    microphone.onclick =
        toggleMicrophone;


    // ---------------------------------------------
    // Close menu when clicking outside
    // ---------------------------------------------

    document.onclick =
        function (event) {

            const menu =
                document.getElementById(
                    "attachment-menu"
                );


            const attach =
                document.getElementById(
                    "attach"
                );


            if (
                menu &&
                attach &&
                !menu.contains(event.target) &&
                !attach.contains(event.target)
            ) {

                closeAttachmentMenu();

            }

        };


    // ---------------------------------------------
    // Update UI
    // ---------------------------------------------

    updateFileDisplay();


    updateMicrophoneButton();


    autoGrowTextarea();


    updateFrameHeight();

}


// ============================================================
// ALWAYS-ON INPUT HANDLERS
//
// Registered once on the document so Enter-to-send, the send
// button and textarea auto-grow keep working even if a Streamlit
// re-render leaves onRender's per-element binding incomplete.
// ============================================================

function isMessageInput(node) {
    return node && node.id === "message";
}

function isInsideSendButton(node) {
    while (node) {
        if (node.id === "send") {
            return true;
        }
        node = node.parentElement;
    }
    return false;
}

document.addEventListener(
    "keydown",
    function (event) {
        if (
            isMessageInput(event.target) &&
            event.key === "Enter" &&
            !event.shiftKey
        ) {
            event.preventDefault();
            submitMessage();
        }
    },
    true
);

document.addEventListener(
    "click",
    function (event) {
        if (isInsideSendButton(event.target)) {
            event.preventDefault();
            submitMessage();
        }
    },
    true
);

document.addEventListener(
    "input",
    function (event) {
        if (isMessageInput(event.target)) {
            autoGrowTextarea();
            updateFrameHeight();
        }
    },
    true
);


// ============================================================
// STREAMLIT SETUP
// ============================================================

Streamlit.events.addEventListener(
    Streamlit.RENDER_EVENT,
    onRender
);


Streamlit.setComponentReady();


Streamlit.setFrameHeight(
    112
);
