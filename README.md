# AgraSynth

A polyphonic synthesizer in C++ with effects, an oscilloscope and MIDI support.

## Features

- 8 voices, sine or saw wave
- Distortion, delay and reverb
- LFO for vibrato or tremolo
- Live oscilloscope
- On-screen piano, computer keyboard and MIDI input
- Record to WAV

## Build

```sh
brew install sdl2 rtmidi
make
./synth
```

## Controls

- `Z`–`M` and `Q`–`U` play notes (the rows above them are the sharps)
- Click the piano keys to play them
- Use the OCT buttons to shift the octave
- Click RECORD to start recording and click it again to save a WAV file
- Press `F11` to toggle fullscreen
