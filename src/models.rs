use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct ClipSegment {
    pub start: String,
    pub end: String,
    pub slow_motion: bool,
}

#[derive(Deserialize, Debug)]
pub struct VideoRequest {
    pub youtube_url: String,
    pub segments: Vec<ClipSegment>,
    pub custom_audio_url: Option<String>,
}
