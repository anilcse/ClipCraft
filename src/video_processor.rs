use std::path::{Path, PathBuf};
use ffmpeg_next as ffmpeg;
use rustube;
use crate::models::{VideoRequest, ClipSegment};
use anyhow::{Result, bail};

pub async fn download_youtube_video(url: &str) -> Result<PathBuf> {
    let yt_video = rustube::download_best_quality(url).await?;
    Ok(yt_video.file_path)
}

pub fn extract_segment(
    input_path: &Path,
    start: &str,
    end: &str,
    output_path: &Path,
) -> Result<()> {
    ffmpeg::init()?;
    let start_ts = parse_time_to_sec(start)?;
    let end_ts = parse_time_to_sec(end)?;
    let duration = end_ts - start_ts;
    if duration <= 0.0 {
        bail!("Invalid segment times");
    }
    // Pseudo-code for trimming
    Ok(())
}

pub fn make_slow_motion(
    input_segment_path: &Path,
    output_path: &Path,
    slow_factor: f64,
) -> Result<()> {
    ffmpeg::init()?;
    // Pseudo-code for slow motion
    Ok(())
}

pub fn concat_segments(segments: &[PathBuf], output_path: &Path) -> Result<()> {
    ffmpeg::init()?;
    // Pseudo-code for concatenation
    Ok(())
}

pub fn replace_audio(
    video_path: &Path,
    audio_path: &Path,
    output_path: &Path
) -> Result<()> {
    ffmpeg::init()?;
    // Pseudo-code for replacing audio
    Ok(())
}

pub async fn process_video_request(req: &VideoRequest) -> Result<PathBuf> {
    let video_path = download_youtube_video(&req.youtube_url).await?;
    let mut all_segments = Vec::new();

    for (i, seg) in req.segments.iter().enumerate() {
        let seg_file = format!("segment_{i}.mp4");
        let seg_path = PathBuf::from(&seg_file);
        extract_segment(&video_path, &seg.start, &seg.end, &seg_path)?;
        all_segments.push(seg_path.clone());

        if seg.slow_motion {
            let slow_file = format!("segment_{i}_slow.mp4");
            let slow_path = PathBuf::from(&slow_file);
            make_slow_motion(&seg_path, &slow_path, 0.5)?;
            all_segments.push(slow_path);
        }
    }

    let final_video_path = PathBuf::from("combined.mp4");
    concat_segments(&all_segments, &final_video_path)?;

    if let Some(url) = &req.custom_audio_url {
        let audio_path = PathBuf::from("custom_audio.mp3");
        let replaced_audio_path = PathBuf::from("final_with_custom_audio.mp4");
        replace_audio(&final_video_path, &audio_path, &replaced_audio_path)?;
        return Ok(replaced_audio_path);
    }

    Ok(final_video_path)
}

fn parse_time_to_sec(t: &str) -> Result<f64> {
    Ok(t.parse().unwrap_or(0.0))
}
