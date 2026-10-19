# Google Drive reports connection

This app searches Google Drive only when a user has connected their account,
has selected one Drive folder, and asks a question about a personal report or
prescription. It receives read-only Drive access and keeps the OAuth token in
process memory only (never written to disk or Chroma). The token is cleared
when the app process stops or the user disconnects.

## One-time Google Cloud setup

1. Create or select a project in the [Google Cloud console](https://console.cloud.google.com/).
2. Enable the **Google Drive API**.
3. Configure the OAuth consent screen.
4. Create an OAuth 2.0 **Web application** client.
5. Add the exact Streamlit URL as an authorized redirect URI. For local use,
   this is normally `http://localhost:8501`.

The redirect URI must match **byte for byte** in three places: the Google
console, `GOOGLE_DRIVE_REDIRECT_URI` in `.env`, and the address you open the
app at. Always open the app as `http://localhost:8501` — `127.0.0.1:8501` is a
different origin to Google and the callback will be rejected.

Add these values to the ignored `.env` file:

```dotenv
GOOGLE_DRIVE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_DRIVE_CLIENT_SECRET=your-client-secret
GOOGLE_DRIVE_REDIRECT_URI=http://localhost:8501
```

Restart both processes after changing configuration:

```bash
cd medical_mcp
python3 server.py
```

```bash
streamlit run app.py
```

## User flow

1. Open the Google Drive sidebar.
2. Paste the URL or ID of the one reports folder to search.
3. Select **Connect Google Drive**, then approve Google's read-only consent
   screen.
4. Ask a report or prescription-related question. The chatbot first uses its
   existing local retrieval, then also searches the selected Google Drive
   folder through the local MCP server.
5. Select **Disconnect Google Drive** at any time. The application removes
   the stored token immediately and also asks Google to revoke it. No Drive
   credentials are written to Chroma or the project files.

Clicking **Continue with Google** navigates the current tab to Google and
back, so the connection survives the redirect. The Drive folder search runs
through the local MCP server, so `python3 server.py` in `medical_mcp` must be
running for personal-report questions to reach Drive.

Only PDF, Google Docs, and plain-text files up to 10 MB are read. Keep the
selected folder limited to the intended person's reports and prescriptions.
