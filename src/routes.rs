use axum::{
    extract::{Json, Multipart},
    response::Html,
    http::StatusCode,
    routing::{get, post},
    Router,
};
use crate::{models::VideoRequest, video_processor::process_video_request};

pub fn create_routes() -> Router {
    Router::new()
        .route("/", get(index))
        .route("/process", post(process_video))
        .route("/upload", post(upload_audio))
}

// A simple GET route returning an HTML form for demonstration
async fn index() -> Html<&'static str> {
    Html(r#"
    <!DOCTYPE html>
    <html>
      <head>
        <title>ClipCraft</title>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@3.2.7/dist/tailwind.min.css" rel="stylesheet">
      </head>
      <body class="p-4">
        <h1 class="text-2xl font-bold mb-4">ClipCraft</h1>
        <form action="/process" method="post" onsubmit="handleSubmit(event)" class="space-y-4">
          <div>
            <label class="block font-medium">YouTube URL</label>
            <input type="text" name="youtube_url" class="border rounded w-full p-1" required />
          </div>
          <div>
            <label class="block font-medium">Segments (JSON Format)</label>
            <textarea name="segments" class="border rounded w-full p-1 h-32">
[
  { "start": "0", "end": "5", "slow_motion": true },
  { "start": "10", "end": "15", "slow_motion": false }
]
            </textarea>
          </div>
          <div>
            <label class="block font-medium">Optional Custom Audio URL</label>
            <input type="text" name="custom_audio_url" class="border rounded w-full p-1" />
          </div>
          <button type="submit" class="bg-blue-500 text-white px-4 py-2 rounded">Process</button>
        </form>
        <hr class="my-4"/>

        <h2 class="text-xl font-bold">Upload an Audio File</h2>
        <form id="uploadForm" action="/upload" method="post" enctype="multipart/form-data" class="space-y-4">
          <div>
            <label class="block font-medium">Choose File</label>
            <input type="file" name="file" />
          </div>
          <button type="submit" class="bg-green-500 text-white px-4 py-2 rounded">Upload</button>
        </form>

        <script>
          async function handleSubmit(e) {
            e.preventDefault();
            const form = e.target;
            const youtube_url = form.youtube_url.value;
            const segments = form.segments.value;
            const custom_audio_url = form.custom_audio_url.value;

            const payload = {
              youtube_url,
              segments: JSON.parse(segments),
              custom_audio_url: custom_audio_url || null
            };

            const res = await fetch('/process', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify(payload)
            });

            if (res.ok) {
              const data = await res.json();
              alert('Video processed! Final path: ' + data.output_path);
            } else {
              alert('Error processing video.');
            }
          }
        </script>
      </body>
    </html>
    "#)
}

// Accept JSON for video editing
async fn process_video(
    Json(payload): Json<VideoRequest>
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    match process_video_request(&payload).await {
        Ok(final_path) => {
            let resp = serde_json::json!({ "output_path": final_path.to_string_lossy() });
            Ok(Json(resp))
        },
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, format!("{e}"))),
    }
}

// Accept multipart file upload (e.g., for custom audio)
async fn upload_audio(mut multipart: Multipart) -> Result<String, (StatusCode, String)> {
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (StatusCode::BAD_REQUEST, format!("Error reading field: {e}"))
    })? {
        let file_name = field
            .file_name()
            .unwrap_or("untitled.bin")
            .to_string();
        let bytes = field.bytes().await.map_err(|e| {
            (StatusCode::BAD_REQUEST, format!("Error reading file bytes: {e}"))
        })?;
        
        // Save the uploaded file to "./uploads/<file_name>"
        tokio::fs::create_dir_all("./uploads").await.map_err(|e| {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Error creating upload dir: {e}"))
        })?;
        tokio::fs::write(format!("./uploads/{}", file_name), &bytes).await.map_err(|e| {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Error writing file: {e}"))
        })?;
    }

    Ok("Upload complete!".to_string())
}
