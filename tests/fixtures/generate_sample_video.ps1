# Regenerates tests/fixtures/sample_video.mp4: a real (tiny) video combining
# a solid-color test pattern with the real speech audio from
# generate_sample_speech.ps1 - real ffmpeg-encoded video, not a mock.
# Run generate_sample_speech.ps1 first if sample_speech.wav doesn't exist yet.
ffmpeg -y -f lavfi -i "color=c=blue:s=320x240:d=4" `
  -i "$PSScriptRoot\sample_speech.wav" `
  -c:v libx264 -c:a aac -shortest "$PSScriptRoot\sample_video.mp4"
