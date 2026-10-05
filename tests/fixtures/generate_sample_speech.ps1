# Regenerates tests/fixtures/sample_speech.wav via Windows SAPI TTS.
# Real synthesized speech, not a mock - used by tests/eval/test_v2_audio_integration.py.
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile("$PSScriptRoot\sample_speech.wav")
$synth.Speak("Decorators are functions that wrap other functions to extend their behavior.")
$synth.Dispose()
