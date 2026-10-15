# Build: make
# Run:   ./synth
# Deps:  brew install sdl2 rtmidi

CXX       := clang++
RTMIDI    := $(shell brew --prefix rtmidi 2>/dev/null)
CXXFLAGS  := -std=c++17 -O2 $(shell sdl2-config --cflags) -I$(RTMIDI)/include
LDFLAGS   := $(shell sdl2-config --libs) \
             -L$(RTMIDI)/lib -lrtmidi \
             -framework CoreMIDI -framework CoreFoundation -framework CoreServices

.PHONY: all clean

all: synth

synth: synth.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f synth
