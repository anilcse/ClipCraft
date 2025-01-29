package main

type ClipSegment struct {
	Start      string `json:"start"`
	End        string `json:"end"`
	SlowMotion bool   `json:"slow_motion"`
}

type VideoRequest struct {
	YouTubeURL     string        `json:"youtube_url"`
	Segments       []ClipSegment `json:"segments"`
	CustomAudioURL *string       `json:"custom_audio_url"`
}
