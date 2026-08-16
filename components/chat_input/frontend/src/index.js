import { Streamlit } from "streamlit-component-lib";

let selectedFiles = [];

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;


// ---------------------------------
// Convert files to Base64
// ---------------------------------

function readFile(file) {

    return new Promise((resolve, reject) => {

        const reader = new FileReader();

        reader.onload = function(event) {

            const base64 =
                event.target.result.split(",")[1];

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


// ---------------------------------
// Send everything to Streamlit
// ---------------------------------

async function submitMessage() {

    const messageInput =
        document.getElementById("message");

    const message =
        messageInput.value.trim();


    // Do not submit empty message
    // unless files are attached

    if (
        !message &&
        selectedFiles.length === 0
    ) {
        return;
    }


    const fileData = [];

    for (const file of selectedFiles) {

        const data =
            await readFile(file);

        fileData.push(data);
    }


    Streamlit.setComponentValue({

        type: "message",

        text: message,

        files: fileData

    });


    messageInput.value = "";

    selectedFiles = [];

    updateFileDisplay();
}


// ---------------------------------
// Update attachment display
// ---------------------------------

function updateFileDisplay() {

    const display =
        document.getElementById("file-display");


    if (!display) {
        return;
    }


    if (selectedFiles.length === 0) {

        display.innerHTML = "";

        return;
    }


    display.innerHTML = "";


    selectedFiles.forEach(
        function(file) {

            const item =
                document.createElement("div");

            item.textContent =
                "📄 " + file.name;

            display.appendChild(item);

        }
    );
}


// ---------------------------------
// Start microphone recording
// ---------------------------------

async function startRecording() {

    try {

        const stream =
            await navigator.mediaDevices.getUserMedia({
                audio: true
            });


        audioChunks = [];


        mediaRecorder =
            new MediaRecorder(stream);


        mediaRecorder.ondataavailable =
            function(event) {

                if (event.data.size > 0) {

                    audioChunks.push(
                        event.data
                    );
                }
            };


        mediaRecorder.onstop =
            async function() {

                const audioBlob =
                    new Blob(
                        audioChunks,
                        {
                            type: "audio/webm"
                        }
                    );


                const reader =
                    new FileReader();


                reader.onloadend =
                    function() {

                        const base64 =
                            reader.result.split(",")[1];


                        Streamlit.setComponentValue({

                            type: "audio",

                            audio: base64,

                            audio_type: "audio/webm"

                        });
                    };


                reader.readAsDataURL(
                    audioBlob
                );


                // Stop microphone

                stream
                    .getTracks()
                    .forEach(
                        track => track.stop()
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


// ---------------------------------
// Stop microphone recording
// ---------------------------------

function stopRecording() {

    if (
        mediaRecorder &&
        mediaRecorder.state !== "inactive"
    ) {

        mediaRecorder.stop();

    }


    isRecording = false;

    updateMicrophoneButton();
}


// ---------------------------------
// Microphone button
// ---------------------------------

function toggleMicrophone() {

    if (isRecording) {

        stopRecording();

    } else {

        startRecording();

    }
}


// ---------------------------------
// Update microphone appearance
// ---------------------------------

function updateMicrophoneButton() {

    const microphone =
        document.getElementById(
            "microphone"
        );


    if (!microphone) {
        return;
    }


    if (isRecording) {

        microphone.textContent = "⏹️";

        microphone.title =
            "Stop recording";

    } else {

        microphone.textContent = "🎙️";

        microphone.title =
            "Voice input";
    }
}

function updateInputFromPython(args) {

    if (!args) {
        return;
    }

    const messageInput =
        document.getElementById("message");

    if (!messageInput) {
        return;
    }

    if (
        args.value !== undefined &&
        args.value !== null
    ) {

        messageInput.value = args.value;

        messageInput.focus();
    }
}

// ---------------------------------
// Component Render
// ---------------------------------

function onRender(event) {

    const args =
        event.detail.args || {};

    updateInputFromPython(args);


    Streamlit.setFrameHeight(
        selectedFiles.length > 0
            ? 120
            : 80
    );


    const sendButton =
        document.getElementById("send");

    const messageInput =
        document.getElementById("message");

    const attachButton =
        document.getElementById("attach");

    const fileInput =
        document.getElementById("fileInput");

    const microphone =
        document.getElementById("microphone");


    if (
        !sendButton ||
        !messageInput ||
        !attachButton ||
        !fileInput ||
        !microphone
    ) {
        return;
    }


    sendButton.onclick =
        submitMessage;


    attachButton.onclick =
        function() {

            fileInput.click();

        };


    fileInput.onchange =
        function() {

            const files =
                Array.from(
                    fileInput.files
                );


            if (files.length > 0) {

                selectedFiles =
                    selectedFiles.concat(
                        files
                    );

                updateFileDisplay();

                Streamlit.setFrameHeight(
                    120
                );
            }


            fileInput.value = "";

        };


    microphone.onclick =
        toggleMicrophone;


    messageInput.onkeydown =
        function(event) {

            if (
                event.key === "Enter"
                && !event.shiftKey
            ) {

                event.preventDefault();

                submitMessage();

            }

        };


    updateMicrophoneButton();
}

// ---------------------------------
// Streamlit Component Setup
// ---------------------------------

Streamlit.events.addEventListener(
    Streamlit.RENDER_EVENT,
    onRender
);


Streamlit.setComponentReady();

Streamlit.setFrameHeight(80);