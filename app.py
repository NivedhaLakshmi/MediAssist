import streamlit as st

from rag import search_local_database

from services.llm_service import (
    OllamaServiceError,
    generate_answer
)

from services.prompt_service import (
    build_healthcare_prompt
)

from mcp_client import (
    search_medical_sources
)

from services.query_utils import (
    normalize_health_query
)

from services.pdf_service import (
    extract_pdf_text
)

from services.patient_memory import (
    add_patient_report,
    search_patient_reports
)

from services.speech_service import (
    transcribe_audio
)

from components.chat_input import (
    chat_input
)

from services.chat_input_service import (
    convert_component_files
)

import base64
import tempfile
import os


MIN_CONTEXT_LENGTH = 200


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Healthcare RAG Assistant",
    page_icon="🏥"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "dictation_text" not in st.session_state:
    st.session_state.dictation_text = ""


if "uploaded_pdf_names" not in st.session_state:
    st.session_state.uploaded_pdf_names = set()


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🏥 Healthcare RAG Assistant")


# ============================================================
# CUSTOM CHAT INPUT CSS
# ============================================================

st.markdown(
    """
    <style>

    iframe[title*="mediassist_chat_input"] {
        position: fixed !important;
        bottom: 20px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: min(900px, 90vw) !important;
        z-index: 999999 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ============================================================
# CUSTOM CHAT INPUT
# ============================================================
#
# Pass dictation_text to the component.
#
# When Whisper produces text, Python stores it in:
#
# st.session_state.dictation_text
#
# The component receives it through:
#
# chat_input(value=...)
#
# ============================================================

chat_result = chat_input(
    value=st.session_state.dictation_text
)


question = ""

uploaded_pdfs = []


# ============================================================
# HANDLE COMPONENT RESULT
# ============================================================

if chat_result:

    result_type = chat_result.get(
        "type",
        "message"
    )


    # ========================================================
    # VOICE INPUT
    # ========================================================

    if result_type == "audio":

        audio_base64 = chat_result.get(
            "audio"
        )


        if audio_base64:

            try:

                # ---------------------------------------------
                # Decode Base64 audio
                # ---------------------------------------------

                audio_bytes = base64.b64decode(
                    audio_base64
                )


                temp_audio_path = None


                # ---------------------------------------------
                # Save temporary audio file
                # ---------------------------------------------

                with tempfile.NamedTemporaryFile(
                    suffix=".webm",
                    delete=False
                ) as temp_audio:

                    temp_audio.write(
                        audio_bytes
                    )

                    temp_audio_path = (
                        temp_audio.name
                    )


                # ---------------------------------------------
                # Transcribe using Whisper
                # ---------------------------------------------

                with st.spinner(
                    "Transcribing..."
                ):

                    question = transcribe_audio(
                        temp_audio_path
                    )


                # ---------------------------------------------
                # Store transcription
                # ---------------------------------------------

                st.session_state.dictation_text = (
                    question
                )


            except Exception as e:

                st.error(
                    "Unable to transcribe the audio."
                )

                print(
                    "Speech-to-text error:"
                )

                print(e)


            finally:

                # ---------------------------------------------
                # Remove temporary audio file
                # ---------------------------------------------

                if (
                    temp_audio_path
                    and os.path.exists(
                        temp_audio_path
                    )
                ):

                    os.remove(
                        temp_audio_path
                    )


            # ---------------------------------------------
            # Rerun so transcription appears
            # in the custom input box
            # ---------------------------------------------

            st.rerun()


    # ========================================================
    # NORMAL MESSAGE
    # ========================================================

    elif result_type == "message":

        # ---------------------------------------------
        # Get question
        # ---------------------------------------------

        question = chat_result.get(
            "text",
            ""
        )


        # ---------------------------------------------
        # Clear transcription after sending
        # ---------------------------------------------

        st.session_state.dictation_text = ""


        # ---------------------------------------------
        # Get uploaded files
        # ---------------------------------------------

        component_files = chat_result.get(
            "files",
            []
        )


        # ---------------------------------------------
        # Convert component files
        # ---------------------------------------------

        if component_files:

            uploaded_pdfs = (
                convert_component_files(
                    component_files
                )
            )


        # ====================================================
        # PROCESS UPLOADED MEDICAL REPORTS
        # ====================================================

        if uploaded_pdfs:

            new_upload_count = 0


            for uploaded_pdf in uploaded_pdfs:

                # ---------------------------------------------
                # Avoid processing same file repeatedly
                # ---------------------------------------------

                if (
                    uploaded_pdf.name
                    not in st.session_state.uploaded_pdf_names
                ):

                    # -----------------------------------------
                    # Extract PDF text
                    # -----------------------------------------

                    pdf_text = extract_pdf_text(
                        uploaded_pdf
                    )


                    # -----------------------------------------
                    # Store in patient database
                    # -----------------------------------------

                    add_patient_report(
                        pdf_text,
                        uploaded_pdf.name,
                        "Medical Report"
                    )


                    # -----------------------------------------
                    # Remember uploaded file
                    # -----------------------------------------

                    st.session_state.uploaded_pdf_names.add(
                        uploaded_pdf.name
                    )


                    new_upload_count += 1


            # ---------------------------------------------
            # Upload success message
            # ---------------------------------------------

            if new_upload_count > 0:

                st.success(
                    f"{new_upload_count} medical "
                    f"report(s) uploaded successfully."
                )


        # ====================================================
        # PROCESS QUESTION
        # ====================================================

        if question:

            # ---------------------------------------------
            # Normalize query
            # ---------------------------------------------

            search_query = normalize_health_query(
                question
            )


            # ---------------------------------------------
            # Store user message
            # ---------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )


            # ---------------------------------------------
            # Display user message
            # ---------------------------------------------

            with st.chat_message("user"):

                st.markdown(
                    question
                )


            # =================================================
            # SEARCH LOCAL DATABASE
            # =================================================

            pdf_vectordb = st.session_state.get(
                "pdf_vectordb"
            )


            docs, local_context, local_source = (
                search_local_database(
                    search_query,
                    pdf_vectordb=pdf_vectordb
                )
            )


            references = []

            context = ""

            source_used = ""


            # =================================================
            # USE LOCAL CONTEXT IF SUFFICIENT
            # =================================================

            if (
                local_context
                and len(local_context.strip())
                > MIN_CONTEXT_LENGTH
            ):

                context = local_context

                source_used = local_source


            else:

                # =============================================
                # SEARCH PATIENT REPORTS
                # =============================================

                (
                    patient_docs,
                    patient_context,
                    patient_source
                ) = search_patient_reports(
                    search_query
                )


                if patient_context:

                    context = patient_context

                    source_used = patient_source

                    docs = patient_docs


                else:

                    # =========================================
                    # SEARCH GENERAL KNOWLEDGE BASE
                    # =========================================

                    (
                        docs,
                        kb_context,
                        kb_source
                    ) = search_local_database(
                        search_query
                    )


                    if (
                        kb_context
                        and len(kb_context.strip())
                        > MIN_CONTEXT_LENGTH
                    ):

                        context = kb_context

                        source_used = kb_source


                    else:

                        # =====================================
                        # SEARCH MCP SOURCES
                        # =====================================

                        (
                            context,
                            source_used,
                            references
                        ) = search_medical_sources(
                            search_query
                        )


            # =================================================
            # DEBUG
            # =================================================

            print(
                f"Using Source: {source_used}"
            )

            print(
                f"Search Query: {search_query}"
            )

            print(
                "========== CONTEXT =========="
            )

            print(
                context
            )

            print(
                "============================="
            )


            # =================================================
            # BUILD PROMPT
            # =================================================

            prompt = build_healthcare_prompt(
                context,
                question
            )


            # =================================================
            # GENERATE ANSWER
            # =================================================

            with st.chat_message("assistant"):

                placeholder = st.empty()


                with st.spinner(
                    "Thinking..."
                ):

                    try:

                        # -------------------------------------
                        # No context
                        # -------------------------------------

                        if not context:

                            answer = (
                                "I could not find enough "
                                "reliable information in the "
                                "local knowledge base, "
                                "MedlinePlus, or PubMed "
                                "for this question."
                            )

                            source_used = (
                                "No reliable source found"
                            )

                            references = []


                        else:

                            # ---------------------------------
                            # Generate LLM response
                            # ---------------------------------

                            answer = generate_answer(
                                prompt
                            )


                        # -------------------------------------
                        # Display answer
                        # -------------------------------------

                        placeholder.markdown(
                            answer
                        )


                        # -------------------------------------
                        # Display source
                        # -------------------------------------

                        st.caption(
                            f"Source Used: {source_used}"
                        )


                    except OllamaServiceError as e:

                        answer = str(e)

                        st.warning(
                            answer
                        )


                    except Exception as e:

                        answer = (
                            "Error generating answer. "
                            "Please check the terminal "
                            "logs for details."
                        )

                        print(
                            "Ollama Error:"
                        )

                        print(
                            e
                        )

                        st.error(
                            answer
                        )


                # =================================================
                # LOCAL KNOWLEDGE BASE SOURCES
                # =================================================

                if (
                    docs
                    and source_used
                    == "Local Knowledge Base"
                ):

                    with st.expander(
                        "Knowledge Base Sources"
                    ):

                        shown = set()


                        for doc in docs:

                            source = doc.metadata.get(
                                "source",
                                "Unknown"
                            )


                            if source not in shown:

                                st.write(
                                    source
                                )

                                shown.add(
                                    source
                                )


                # =================================================
                # MEDLINEPLUS REFERENCES
                # =================================================

                if (
                    source_used
                    == "MedlinePlus"
                    and references
                ):

                    with st.expander(
                        "MedlinePlus References"
                    ):

                        for topic in references:

                            st.write(
                                f"• [{topic['title']}]"
                                f"({topic['url']})"
                            )


                # =================================================
                # PUBMED REFERENCES
                # =================================================

                if (
                    source_used
                    == "PubMed"
                    and references
                ):

                    with st.expander(
                        "PubMed References"
                    ):

                        for paper in references:

                            st.write(
                                f"• {paper['title']}"
                            )


                # =================================================
                # CLINICAL TRIALS REFERENCES
                # =================================================

                if (
                    source_used
                    == "ClinicalTrials.gov"
                    and references
                ):

                    with st.expander(
                        "Clinical Trials"
                    ):

                        for trial in references:

                            st.write(
                                f"• [{trial['title']}]"
                                f"({trial['url']})"
                            )


            # =================================================
            # STORE ASSISTANT MESSAGE
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )