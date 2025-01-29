package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/3d0c/gmf"
	youtube "github.com/kkdai/youtube/v2"
)

// Download a YouTube video using the youtube/v2 library.
func downloadYouTubeVideo(url string) (string, error) {
	client := youtube.Client{}
	video, err := client.GetVideo(url)
	if err != nil {
		return "", err
	}

	// Attempt to find a format that has audio
	var best *youtube.Format
	for _, f := range video.Formats {
		if f.AudioChannels > 0 {
			best = &f
			break
		}
	}
	if best == nil {
		return "", errors.New("no format with audio channels found")
	}

	stream, _, err := client.GetStream(video, best)
	if err != nil {
		return "", err
	}
	defer stream.Close()

	outName := "youtube_video.mp4"
	outFile, err := os.Create(outName)
	if err != nil {
		return "", err
	}
	defer outFile.Close()

	_, err = io.Copy(outFile, stream)
	if err != nil {
		return "", err
	}

	return outName, nil
}

// Stub for extracting a segment [start..end]
func extractSegment(inputPath, start, end, outputPath string) error {
	// For real usage, you'd open the file with gmf, and slice frames.
	sStart, _ := parseTimeToSec(start)
	sEnd, _ := parseTimeToSec(end)
	if sEnd <= sStart {
		return errors.New("invalid segment times")
	}

	// Example: do nothing, pretend success
	return nil
}

// Stub for producing a slow motion version
func makeSlowMotion(inputPath, outputPath string, slowFactor float64) error {
	// For real usage, you'd set filters in gmf to slow down playback
	return nil
}

// Stub for concatenating segments
func concatSegments(paths []string, outputPath string) error {
	// For real usage, you'd do a concat demuxer approach or read/write frames
	return nil
}

// Stub for replacing audio
func replaceAudio(videoPath, audioPath, outputPath string) error {
	// For real usage, you'd open both with gmf, strip the old audio track, etc.
	return nil
}

// Orchestrate all steps
func ProcessVideoRequest(req VideoRequest) (string, error) {
	// 1. Download YouTube
	videoPath, err := downloadYouTubeVideo(req.YouTubeURL)
	if err != nil {
		return "", err
	}

	// 2. Extract segments
	var segmentsPaths []string
	for i, seg := range req.Segments {
		segFile := fmt.Sprintf("segment_%d.mp4", i)
		err := extractSegment(videoPath, seg.Start, seg.End, segFile)
		if err != nil {
			return "", err
		}
		segmentsPaths = append(segmentsPaths, segFile)

		// If slow motion
		if seg.SlowMotion {
			slowFile := fmt.Sprintf("segment_%d_slow.mp4", i)
			err := makeSlowMotion(segFile, slowFile, 0.5)
			if err != nil {
				return "", err
			}
			segmentsPaths = append(segmentsPaths, slowFile)
		}
	}

	// 3. Concatenate them
	finalPath := "combined.mp4"
	err = concatSegments(segmentsPaths, finalPath)
	if err != nil {
		return "", err
	}

	// 4. Replace audio if provided
	if req.CustomAudioURL != nil {
		audioPath := "custom_audio.mp3"
		// You could download that from the URL to audioPath
		finalWithAudio := "final_with_custom_audio.mp4"
		err = replaceAudio(finalPath, audioPath, finalWithAudio)
		if err != nil {
			return "", err
		}
		return finalWithAudio, nil
	}

	return finalPath, nil
}

func parseTimeToSec(str string) (float64, error) {
	return strconv.ParseFloat(str, 64)
}

func init() {
	// If your gmf usage needed initialization:
	_ = gmf.InitFFmpeg()
}
