#include <SDL.h>
#include <rtmidi/RtMidi.h>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <algorithm>
#include <vector>
#include <string>

static const int   WIN_W   = 1000;
static const int   WIN_H   = 560;
static const int   SR      = 44100;
static const float TWO_PI  = 6.283185307f;
static const int   OSC     = 1024;
static const int   REC_MAX = SR * 120;
static const int   VOICES  = 8;

static const int PIANO_Y = 392;
static const int PIANO_X = 17;
static const int WHITE_W = 46;
static const int WHITE_H = 158;
static const int BLACK_W = 28;
static const int BLACK_H = 100;

// ── Voice ─────────────────────────────────────────────────────────────────────
struct Voice {
    float    freq   = 0.f;
    float    phase  = 0.f;
    float    env    = 0.f;
    float    target = 0.f;
    int      note   = -1;
    bool     used   = false;
    uint32_t age    = 0;
};
static Voice    g_voices[VOICES] = {};
static uint32_t g_voiceGen       = 0;
static SDL_AudioDeviceID g_dev   = 0;

// ── Synth params (atomic) ─────────────────────────────────────────────────────
static std::atomic<float> g_vol      { 0.5f };
static std::atomic<float> g_dist     { 0.f  };
static std::atomic<int>   g_wave     { 0    };
static std::atomic<float> g_dlyTime  { 0.3f };
static std::atomic<float> g_dlyMix   { 0.f  };
static std::atomic<float> g_reverb   { 0.f  };
static std::atomic<float> g_lfoRate  { 5.f  };
static std::atomic<float> g_lfoDepth { 0.f  };
static std::atomic<int>   g_lfoTgt   { 0    }; // 0=vibrato 1=tremolo
static std::atomic<bool>  g_lfoOn    { false };

// ── Oscilloscope ──────────────────────────────────────────────────────────────
static float            g_oscBuf[OSC] = {};
static std::atomic<int> g_oscPos { 0 };

// ── Recording ─────────────────────────────────────────────────────────────────
static float*            g_recBuf    = nullptr;
static std::atomic<int>  g_recPos    { 0 };
static std::atomic<bool> g_recording { false };

// ── Delay ─────────────────────────────────────────────────────────────────────
static const int DELAY_MAX = SR;
static float g_delayBuf[DELAY_MAX] = {};
static int   g_delayWr = 0;

// ── Reverb ────────────────────────────────────────────────────────────────────
static const int COMB_SZ[4] = { 1116, 1188, 1277, 1356 };
static float g_comb[4][1400] = {};
static int   g_combPos[4] = {};
static float g_combLP[4]  = {};
static const int AP_SZ[2] = { 556, 441 };
static float g_ap[2][600] = {};
static int   g_apPos[2]   = {};

// ── LFO (audio-thread only) ───────────────────────────────────────────────────
static float g_lfoPhase   = 0.f;
static float g_voiceNorm  = 1.f; // smoothed 1/activeVoices to prevent polyphonic clipping

// ── Piano key highlight state (written from main+MIDI threads, read for draw) ─
static bool g_keyDisplay[128] = {};

// ── Note helpers ──────────────────────────────────────────────────────────────
static float noteHz(int midi) { return 440.f * std::pow(2.f, (midi - 69) / 12.f); }

static void noteOn(int note) {
    if (note < 0 || note > 127) return;
    SDL_LockAudioDevice(g_dev);
    int idx = -1;
    uint32_t oldest = UINT32_MAX;
    for (int i = 0; i < VOICES; i++) {
        if (!g_voices[i].used) { idx = i; oldest = 0; break; }
        if (g_voices[i].age < oldest) { oldest = g_voices[i].age; idx = i; }
    }
    if (idx < 0) idx = 0;
    // Preserve phase/env of stolen voice to reduce clicks
    g_voices[idx].freq   = noteHz(note);
    g_voices[idx].target = 1.f;
    g_voices[idx].note   = note;
    g_voices[idx].used   = true;
    g_voices[idx].age    = ++g_voiceGen;
    SDL_UnlockAudioDevice(g_dev);
}

static void noteOff(int note) {
    if (note < 0 || note > 127) return;
    SDL_LockAudioDevice(g_dev);
    for (int i = 0; i < VOICES; i++)
        if (g_voices[i].note == note && g_voices[i].used && g_voices[i].target > 0.5f)
            g_voices[i].target = 0.f;
    SDL_UnlockAudioDevice(g_dev);
}

// Polynomial band-limited step: smooths the sawtooth discontinuity to kill aliasing.
static float polyBlep(float t, float dt) {
    if (t < dt)       { t /= dt;        return  t+t - t*t - 1.f; }
    if (t > 1.f - dt) { t = (t-1.f)/dt; return  t*t + t+t + 1.f; }
    return 0.f;
}

// ── Audio callback ────────────────────────────────────────────────────────────
static void audioCB(void*, Uint8* stream, int len)
{
    float* out = reinterpret_cast<float*>(stream);
    int n = len / sizeof(float);

    float vol      = g_vol.load();
    float dist     = g_dist.load();
    int   wave     = g_wave.load();
    float dlyMix   = g_dlyMix.load();
    float rvbMix   = g_reverb.load();
    float lfoRate  = g_lfoRate.load();
    float lfoDepth = g_lfoDepth.load();
    int   lfoTgt   = g_lfoTgt.load();
    bool  lfoOn    = g_lfoOn.load();
    int   dlyN     = std::max(1, (int)(g_dlyTime.load() * 0.8f * SR));

    const float atkC = 1.f - std::exp(-1.f / (0.003f * SR));  // 3 ms attack
    const float relC = 1.f - std::exp(-1.f / (0.120f * SR));  // 120 ms release

    // Normalize by total envelope energy so released voices don't suppress
    // the level of held notes as they decay — tracks actual loudness, not count.
    float envSum = 0.f;
    for (int v = 0; v < VOICES; v++) if (g_voices[v].used) envSum += g_voices[v].env;
    float targetNorm = 1.f / std::max(1.f, envSum);
    const float normC = 1.f - std::exp(-1.f / (0.005f * SR)); // 5ms smoothing

    for (int i = 0; i < n; ++i) {
        // LFO
        float lfo = std::sin(g_lfoPhase * TWO_PI);
        g_lfoPhase += lfoRate / SR;
        if (g_lfoPhase >= 1.f) g_lfoPhase -= 1.f;

        float s = 0.f;
        for (int v = 0; v < VOICES; v++) {
            Voice& vc = g_voices[v];
            if (!vc.used) continue;

            float coef = (vc.target > vc.env) ? atkC : relC;
            vc.env += coef * (vc.target - vc.env);
            if (vc.target < 0.5f && vc.env < 0.0001f) {
                vc.used = false; vc.env = 0.f; continue;
            }

            // Vibrato: pitch modulation
            float freq = vc.freq;
            if (lfoOn && lfoTgt == 0) freq *= std::exp2(lfoDepth * 0.04f * lfo);

            float inc = freq / SR;
            float samp;
            if (wave == 0) {
                samp = std::sin(vc.phase * TWO_PI);
            } else {
                samp = 2.f * vc.phase - 1.f - polyBlep(vc.phase, inc);
            }
            vc.phase += inc;
            if (vc.phase >= 1.f) vc.phase -= 1.f;

            // Tremolo: amplitude modulation
            float amp = vc.env;
            if (lfoOn && lfoTgt == 1) amp *= 1.f - lfoDepth * 0.5f * (1.f - lfo);

            s += samp * amp;
        }

        g_voiceNorm += normC * (targetNorm - g_voiceNorm);
        s = s * g_voiceNorm * vol;

        if (dist > 0.001f) {
            float drv = 1.f + dist * 20.f;
            s = std::tanh(s * drv) / std::tanh(drv);
        }

        { int p = g_oscPos.load(std::memory_order_relaxed);
          g_oscBuf[p] = s;
          g_oscPos.store((p+1) % OSC, std::memory_order_release); }

        { int rp = (g_delayWr - dlyN + DELAY_MAX) % DELAY_MAX;
          float echo = g_delayBuf[rp];
          g_delayBuf[g_delayWr] = s + echo * 0.4f;
          g_delayWr = (g_delayWr + 1) % DELAY_MAX;
          s += echo * dlyMix; }

        { float rvbIn = s * 0.04f, rvbOut = 0.f;
          for (int c = 0; c < 4; ++c) {
              float co = g_comb[c][g_combPos[c]];
              g_combLP[c] = co * 0.5f + g_combLP[c] * 0.5f;
              g_comb[c][g_combPos[c]] = rvbIn + g_combLP[c] * 0.90f;
              g_combPos[c] = (g_combPos[c] + 1) % COMB_SZ[c];
              rvbOut += co;
          }
          for (int a = 0; a < 2; ++a) {
              float ao = g_ap[a][g_apPos[a]];
              g_ap[a][g_apPos[a]] = rvbOut + ao * 0.5f;
              g_apPos[a] = (g_apPos[a] + 1) % AP_SZ[a];
              rvbOut = ao - rvbOut;
          }
          s += rvbOut * rvbMix * 0.4f; }

        out[i] = std::max(-1.f, std::min(1.f, s));

        if (g_recording.load(std::memory_order_relaxed)) {
            int rp = g_recPos.load(std::memory_order_relaxed);
            if (rp < REC_MAX) { g_recBuf[rp] = out[i]; g_recPos.store(rp+1, std::memory_order_relaxed); }
            else { g_recording.store(false, std::memory_order_release); }
        }
    }
}

// ── MIDI callback (runs on MIDI thread) ───────────────────────────────────────
static void midiCB(double, std::vector<unsigned char>* msg, void*)
{
    if (!msg || msg->size() < 2) return;
    unsigned char status = (*msg)[0] & 0xF0;
    int note = (*msg)[1];
    int vel  = (msg->size() >= 3) ? (*msg)[2] : 64;
    if (status == 0x90 && vel > 0) {
        noteOn(note);
        if (note >= 0 && note < 128) g_keyDisplay[note] = true;
    } else if (status == 0x80 || (status == 0x90 && vel == 0)) {
        noteOff(note);
        if (note >= 0 && note < 128) g_keyDisplay[note] = false;
    }
}

// ── Font ──────────────────────────────────────────────────────────────────────
static const uint8_t FONT[59][8] = {
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x18,0x3C,0x3C,0x18,0x18,0x00,0x18,0x00},
    {0x36,0x36,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x36,0x36,0x7F,0x36,0x7F,0x36,0x36,0x00},
    {0x0C,0x3E,0x03,0x1E,0x30,0x1F,0x0C,0x00},
    {0x00,0x63,0x33,0x18,0x0C,0x66,0x63,0x00},
    {0x1C,0x36,0x1C,0x6E,0x3B,0x33,0x6E,0x00},
    {0x06,0x06,0x03,0x00,0x00,0x00,0x00,0x00},
    {0x18,0x0C,0x06,0x06,0x06,0x0C,0x18,0x00},
    {0x06,0x0C,0x18,0x18,0x18,0x0C,0x06,0x00},
    {0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00},
    {0x00,0x0C,0x0C,0x3F,0x0C,0x0C,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x0C,0x06,0x00},
    {0x00,0x00,0x00,0x3F,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C,0x00},
    {0x00,0x60,0x30,0x18,0x0C,0x06,0x00,0x00},
    {0x3E,0x63,0x73,0x7B,0x6F,0x67,0x3E,0x00},
    {0x0C,0x0E,0x0C,0x0C,0x0C,0x0C,0x3F,0x00},
    {0x1E,0x33,0x30,0x1C,0x06,0x33,0x3F,0x00},
    {0x1E,0x33,0x30,0x1C,0x30,0x33,0x1E,0x00},
    {0x38,0x3C,0x36,0x33,0x7F,0x30,0x78,0x00},
    {0x3F,0x03,0x1F,0x30,0x30,0x33,0x1E,0x00},
    {0x1C,0x06,0x03,0x1F,0x33,0x33,0x1E,0x00},
    {0x3F,0x33,0x30,0x18,0x0C,0x06,0x06,0x00},
    {0x1E,0x33,0x33,0x1E,0x33,0x33,0x1E,0x00},
    {0x1E,0x33,0x33,0x3E,0x30,0x18,0x0E,0x00},
    {0x00,0x0C,0x0C,0x00,0x00,0x0C,0x0C,0x00},
    {0x00,0x0C,0x0C,0x00,0x00,0x0C,0x0C,0x06},
    {0x18,0x0C,0x06,0x03,0x06,0x0C,0x18,0x00},
    {0x00,0x00,0x3F,0x00,0x3F,0x00,0x00,0x00},
    {0x06,0x0C,0x18,0x30,0x18,0x0C,0x06,0x00},
    {0x1E,0x33,0x30,0x18,0x0C,0x00,0x0C,0x00},
    {0x3E,0x63,0x7B,0x7B,0x7B,0x03,0x1E,0x00},
    {0x0C,0x1E,0x33,0x33,0x3F,0x33,0x33,0x00},
    {0x3F,0x66,0x66,0x3E,0x66,0x66,0x3F,0x00},
    {0x3C,0x66,0x03,0x03,0x03,0x66,0x3C,0x00},
    {0x1F,0x36,0x66,0x66,0x66,0x36,0x1F,0x00},
    {0x7F,0x46,0x16,0x1E,0x16,0x46,0x7F,0x00},
    {0x7F,0x46,0x16,0x1E,0x16,0x06,0x0F,0x00},
    {0x3C,0x66,0x03,0x03,0x73,0x66,0x7C,0x00},
    {0x33,0x33,0x33,0x3F,0x33,0x33,0x33,0x00},
    {0x1E,0x0C,0x0C,0x0C,0x0C,0x0C,0x1E,0x00},
    {0x78,0x30,0x30,0x30,0x33,0x33,0x1E,0x00},
    {0x67,0x66,0x36,0x1E,0x36,0x66,0x67,0x00},
    {0x0F,0x06,0x06,0x06,0x46,0x66,0x7F,0x00},
    {0x63,0x77,0x7F,0x7F,0x6B,0x63,0x63,0x00},
    {0x63,0x67,0x6F,0x7B,0x73,0x63,0x63,0x00},
    {0x1C,0x36,0x63,0x63,0x63,0x36,0x1C,0x00},
    {0x3F,0x66,0x66,0x3E,0x06,0x06,0x0F,0x00},
    {0x1E,0x33,0x33,0x33,0x3B,0x1E,0x38,0x00},
    {0x3F,0x66,0x66,0x3E,0x36,0x66,0x67,0x00},
    {0x1E,0x33,0x07,0x0E,0x38,0x33,0x1E,0x00},
    {0x3F,0x2D,0x0C,0x0C,0x0C,0x0C,0x1E,0x00},
    {0x33,0x33,0x33,0x33,0x33,0x33,0x3F,0x00},
    {0x33,0x33,0x33,0x33,0x33,0x1E,0x0C,0x00},
    {0x63,0x63,0x63,0x6B,0x7F,0x77,0x63,0x00},
    {0x63,0x63,0x36,0x1C,0x1C,0x36,0x63,0x00},
    {0x33,0x33,0x33,0x1E,0x0C,0x0C,0x1E,0x00},
    {0x7F,0x63,0x31,0x18,0x4C,0x66,0x7F,0x00},
};

// ── Draw helpers ──────────────────────────────────────────────────────────────
static SDL_Renderer* ren = nullptr;
static SDL_Window*   gWin = nullptr;

// Convert window-space mouse coords to logical 1000x560 coords.
// SDL_RenderWindowToLogical mishandles macOS DPI in fullscreen; we do it manually.
static void toLogical(int wx, int wy, int& lx, int& ly) {
    int rw, rh, ww, wh;
    SDL_GetRendererOutputSize(ren, &rw, &rh);
    SDL_GetWindowSize(gWin, &ww, &wh);
    float px = wx * (float)rw / ww;
    float py = wy * (float)rh / wh;
    float scale = std::min((float)rw / WIN_W, (float)rh / WIN_H);
    lx = (int)((px - (rw - WIN_W * scale) * 0.5f) / scale);
    ly = (int)((py - (rh - WIN_H * scale) * 0.5f) / scale);
}
struct C4 { Uint8 r,g,b,a=255; };

static void setCol(C4 c) { SDL_SetRenderDrawColor(ren,c.r,c.g,c.b,c.a); }
static void fillR(int x,int y,int w,int h,C4 c){setCol(c);SDL_Rect r{x,y,w,h};SDL_RenderFillRect(ren,&r);}
static void wireR(int x,int y,int w,int h,C4 c){setCol(c);SDL_Rect r{x,y,w,h};SDL_RenderDrawRect(ren,&r);}

static void drawChar(int x,int y,char ch,int sc,C4 col){
    int ci=(unsigned char)ch-0x20; if(ci<0||ci>58)return;
    for(int row=0;row<8;++row){uint8_t b=FONT[ci][row];
        for(int bit=0;bit<8;++bit) if(b&(1<<bit)) fillR(x+bit*sc,y+row*sc,sc,sc,col);}
}
static int  strPx(const char*s,int sc){int n=(int)strlen(s);return n==0?0:(n-1)*9*sc+8*sc;}
static void drawStr(int x,int y,const char*s,int sc,C4 col){for(int i=0;s[i];++i)drawChar(x+i*9*sc,y,s[i],sc,col);}
static void drawStrC(int cx,int y,const char*s,int sc,C4 col){drawStr(cx-strPx(s,sc)/2,y,s,sc,col);}

// ── Slider ────────────────────────────────────────────────────────────────────
struct Slider {
    int x,y,w,h; float*val,lo,hi; const char*label;
    int fmt=0; // 0=percent  1=ms  2=hz
    bool drag=false;
    int handleX()const{return x+(int)((*val-lo)/(hi-lo)*(w-12));}
    void draw()const{
        drawStrC(x+w/2,y-16,label,1,{100,92,138});
        int hx=handleX();
        fillR(x,y+h/2-3,w,6,{35,30,55}); fillR(x,y+h/2-3,hx-x+6,6,{105,65,185});
        SDL_Rect hnd{hx,y+h/2-10,12,20};
        SDL_SetRenderDrawColor(ren,210,170,255,255); SDL_RenderFillRect(ren,&hnd);
        SDL_SetRenderDrawColor(ren,255,230,255,255); SDL_RenderDrawRect(ren,&hnd);
        char buf[24];
        if(fmt==1) std::snprintf(buf,sizeof(buf),"%.0fms",*val*800.f);
        else if(fmt==2) std::snprintf(buf,sizeof(buf),"%.1fHz",*val);
        else std::snprintf(buf,sizeof(buf),"%.0f%%",*val*100.f);
        drawStrC(x+w/2,y+h+4,buf,1,{72,65,100});
    }
    void onDown(int mx,int my){if(mx>=x&&mx<x+w&&my>=y&&my<y+h){*val=lo+std::clamp(float(mx-x)/w,0.f,1.f)*(hi-lo);drag=true;}}
    void onUp(){drag=false;}
    void onMove(int mx){if(!drag)return;*val=lo+std::clamp(float(mx-x)/w,0.f,1.f)*(hi-lo);}
};

// ── Oscilloscope ──────────────────────────────────────────────────────────────
static void drawOsc(int x,int y,int w,int h,float dist){
    fillR(x,y,w,h,{12,10,22}); wireR(x,y,w,h,{55,44,88});
    int cy=y+h/2,half=h/2-8;
    SDL_SetRenderDrawColor(ren,28,24,46,255);
    SDL_RenderDrawLine(ren,x,cy,x+w,cy);
    Uint8 lr=(Uint8)(80+(int)(dist*175)),lg=(Uint8)(255-(int)(dist*155)),lb=(Uint8)(120-(int)(dist*70));
    int head=g_oscPos.load(std::memory_order_acquire),start=(head-w+OSC)%OSC,px=-1,py=-1;
    SDL_SetRenderDrawBlendMode(ren,SDL_BLENDMODE_BLEND);
    for(int i=0;i<w;++i){
        int sy=cy-(int)(g_oscBuf[(start+i)%OSC]*half);
        if(px>=0){
            SDL_SetRenderDrawColor(ren,lr,lg,lb,45);
            SDL_RenderDrawLine(ren,px,py-1,x+i,sy-1);
            SDL_RenderDrawLine(ren,px,py+1,x+i,sy+1);
            SDL_SetRenderDrawColor(ren,lr,lg,lb,255);
            SDL_RenderDrawLine(ren,px,py,x+i,sy);
        }
        px=x+i;py=sy;
    }
    SDL_SetRenderDrawBlendMode(ren,SDL_BLENDMODE_NONE);
}

// ── Piano ─────────────────────────────────────────────────────────────────────
static const int wMidi[21]={48,50,52,53,55,57,59,60,62,64,65,67,69,71,72,74,76,77,79,81,83};
static const float bOff[15]={0.65f,1.65f,3.65f,4.65f,5.65f,7.65f,8.65f,10.65f,11.65f,12.65f,14.65f,15.65f,17.65f,18.65f,19.65f};
static const int bMidi[15]={49,51,54,56,58,61,63,66,68,70,73,75,78,80,82};
static const char*wLabel[21]={"Z","X","C","V","B","N","M","Q","W","E","R","T","Y","U","C5","D5","E5","F5","G5","A5","B5"};

static void drawPiano(const bool* active){
    for(int i=0;i<21;++i){
        int kx=PIANO_X+i*WHITE_W; bool on=active[wMidi[i]];
        fillR(kx,PIANO_Y,WHITE_W,WHITE_H,on?C4{123,82,255}:C4{212,206,228});
        if(!on){SDL_SetRenderDrawColor(ren,240,236,255,255);SDL_RenderDrawLine(ren,kx+1,PIANO_Y+1,kx+WHITE_W-2,PIANO_Y+1);}
        wireR(kx,PIANO_Y,WHITE_W,WHITE_H,{75,65,105});
        drawStrC(kx+WHITE_W/2,PIANO_Y+WHITE_H-13,wLabel[i],1,on?C4{220,200,255}:C4{100,90,135});
    }
    for(int i=0;i<15;++i){
        int kx=PIANO_X+(int)(bOff[i]*WHITE_W)-BLACK_W/2; bool on=active[bMidi[i]];
        fillR(kx,PIANO_Y,BLACK_W,BLACK_H,on?C4{123,82,255}:C4{18,14,28});
        wireR(kx,PIANO_Y,BLACK_W,BLACK_H,{50,44,72});
    }
}

static int pianoHit(int mx,int my){
    if(my<PIANO_Y||my>=PIANO_Y+WHITE_H||mx<PIANO_X||mx>=PIANO_X+21*WHITE_W)return -1;
    if(my<PIANO_Y+BLACK_H)for(int i=0;i<15;++i){int kx=PIANO_X+(int)(bOff[i]*WHITE_W)-BLACK_W/2;if(mx>=kx&&mx<kx+BLACK_W)return bMidi[i];}
    return wMidi[(mx-PIANO_X)/WHITE_W];
}

static int keyToNote(SDL_Keycode k){
    switch(k){
    case SDLK_z:return 48;case SDLK_s:return 49;case SDLK_x:return 50;case SDLK_d:return 51;
    case SDLK_c:return 52;case SDLK_v:return 53;case SDLK_g:return 54;case SDLK_b:return 55;
    case SDLK_h:return 56;case SDLK_n:return 57;case SDLK_j:return 58;case SDLK_m:return 59;
    case SDLK_q:return 60;case SDLK_2:return 61;case SDLK_w:return 62;case SDLK_3:return 63;
    case SDLK_e:return 64;case SDLK_r:return 65;case SDLK_5:return 66;case SDLK_t:return 67;
    case SDLK_6:return 68;case SDLK_y:return 69;case SDLK_7:return 70;case SDLK_u:return 71;
    default:return -1;}
}

// ── WAV save ──────────────────────────────────────────────────────────────────
static void writeWav(const char*path,const float*buf,int n){
    FILE*f=fopen(path,"wb"); if(!f)return;
    auto w16=[&](int16_t v){fwrite(&v,2,1,f);};
    auto w32=[&](int32_t v){fwrite(&v,4,1,f);};
    auto wcc=[&](const char*s){fwrite(s,1,4,f);};
    int32_t dataBytes=n*2;
    wcc("RIFF");w32(36+dataBytes);wcc("WAVE");
    wcc("fmt ");w32(16);w16(1);w16(1);w32(SR);w32(SR*2);w16(2);w16(16);
    wcc("data");w32(dataBytes);
    for(int i=0;i<n;i++){float s=std::max(-1.f,std::min(1.f,buf[i]));int16_t v=(int16_t)(s*32767.f);fwrite(&v,2,1,f);}
    fclose(f);
}

static void chooseFolder(char*out,int sz){
    FILE*pipe=popen("osascript -e 'POSIX path of (choose folder with prompt \"Choose save folder:\")'","r");
    if(!pipe)return;
    char buf[512]={};
    if(fgets(buf,sizeof(buf),pipe)){int l=(int)strlen(buf);while(l>0&&(buf[l-1]=='\n'||buf[l-1]=='\r'))buf[--l]='\0';strncpy(out,buf,sz-1);}
    pclose(pipe);
}

static void doSave(const char*folder,char*lastSaved,int lsz){
    int n=g_recPos.load(std::memory_order_acquire); if(n==0)return;
    time_t t=time(nullptr); struct tm*ti=localtime(&t);
    char fname[64]; strftime(fname,sizeof(fname),"recording_%Y%m%d_%H%M%S.wav",ti);
    char path[640];
    if(folder[0]){int fl=(int)strlen(folder);snprintf(path,sizeof(path),"%s%s%s",folder,(fl>0&&folder[fl-1]=='/')?"":"/",fname);}
    else{const char*h=getenv("HOME");snprintf(path,sizeof(path),"%s/Desktop/%s",h?h:"/tmp",fname);}
    writeWav(path,g_recBuf,n);
    strncpy(lastSaved,fname,lsz-1);lastSaved[lsz-1]='\0';
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int,char**){
    g_recBuf = new float[REC_MAX];

    SDL_Init(SDL_INIT_VIDEO|SDL_INIT_AUDIO);
    SDL_Window*win=SDL_CreateWindow("Synth",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,WIN_W,WIN_H,SDL_WINDOW_SHOWN|SDL_WINDOW_RESIZABLE);
    gWin=win;
    ren=SDL_CreateRenderer(win,-1,SDL_RENDERER_ACCELERATED|SDL_RENDERER_PRESENTVSYNC);
    SDL_RenderSetLogicalSize(ren,WIN_W,WIN_H);

    SDL_AudioSpec want{},got{};
    want.freq=SR;want.format=AUDIO_F32SYS;want.channels=1;want.samples=512;want.callback=audioCB;
    g_dev=SDL_OpenAudioDevice(nullptr,0,&want,&got,0);
    SDL_PauseAudioDevice(g_dev,0);

    // ── MIDI setup ────────────────────────────────────────────────────────────
    RtMidiIn* midiIn = nullptr;
    std::string midiPortName = "NONE";
    try {
        midiIn = new RtMidiIn();
        if (midiIn->getPortCount() > 0) {
            midiIn->openPort(0);
            midiPortName = midiIn->getPortName(0);
            // Truncate for display
            if (midiPortName.size() > 18) midiPortName = midiPortName.substr(0, 17) + ".";
            midiIn->setCallback(midiCB);
            midiIn->ignoreTypes(false, false, false);
        }
    } catch (...) {
        delete midiIn;
        midiIn = nullptr;
    }

    float volVal      = 0.5f;
    float distVal     = 0.f;
    float dlyTime     = 0.3f;
    float dlyMix      = 0.f;
    float rvbVal      = 0.f;
    float lfoRateVal  = 5.f;
    float lfoDepthVal = 0.f;
    int   wave        = 0;
    int   octaveShift = 0;
    int   lfoTarget   = 0;
    bool  lfoOn       = false;
    bool  playDrone   = false;
    bool  recording   = false;
    char  saveFolder[512] = {};
    char  lastSaved[128]  = {};

    Slider sliders[7] = {
        {300,170,126,42,&volVal,      0.f, 1.f,  "VOLUME",     0},
        {438,170,126,42,&distVal,     0.f, 1.f,  "DISTORTION", 0},
        {576,170,126,42,&dlyTime,     0.f, 1.f,  "DELAY TIME", 1},
        {714,170,126,42,&dlyMix,      0.f, 1.f,  "DELAY MIX",  0},
        {852,170,126,42,&rvbVal,      0.f, 1.f,  "REVERB",     0},
        {300,248,170,38,&lfoRateVal,  0.1f,20.f, "LFO RATE",   2},
        {490,248,170,38,&lfoDepthVal, 0.f, 1.f,  "LFO DEPTH",  0},
    };

    // keyHeld: indexed by base note (no octave shift), for keyboard keys
    bool keyHeld[128]       = {};
    int  keyActualNote[128] = {};
    int  mouseBase   = -1;
    int  mouseActual = -1;
    bool fullscreen  = false;
    bool running = true;
    SDL_Event ev;

    while(running){
        while(SDL_PollEvent(&ev)){
            switch(ev.type){
            case SDL_QUIT: running=false; break;

            case SDL_MOUSEBUTTONDOWN:
                if(ev.button.button!=SDL_BUTTON_LEFT)break;
                {
                int mx=ev.button.x,my=ev.button.y;
                for(auto&sl:sliders)sl.onDown(mx,my);

                // Waveform
                if(mx>=20 &&mx<150&&my>=170&&my<212)wave=0;
                if(mx>=158&&mx<288&&my>=170&&my<212)wave=1;

                // LFO target
                if(mx>=20 &&mx<160&&my>=248&&my<286)lfoOn=!lfoOn;
                if(mx>=680&&mx<800&&my>=248&&my<286)lfoTarget=0;
                if(mx>=812&&mx<932&&my>=248&&my<286)lfoTarget=1;

                // Octave
                if(mx>=22&&mx<66&&my>=316&&my<350)  octaveShift=std::max(-2,octaveShift-1);
                if(mx>=152&&mx<196&&my>=316&&my<350) octaveShift=std::min(2,octaveShift+1);

                // Play drone (A4)
                if(mx>=230&&mx<400&&my>=306&&my<358){
                    if(!playDrone){ noteOn(60); g_keyDisplay[60]=true;  playDrone=true;  }
                    else          { noteOff(60); g_keyDisplay[60]=false; playDrone=false; }
                }

                // Record
                if(mx>=416&&mx<584&&my>=306&&my<358){
                    if(!recording){g_recPos.store(0);g_recording.store(true);recording=true;}
                    else{g_recording.store(false,std::memory_order_release);recording=false;doSave(saveFolder,lastSaved,sizeof(lastSaved));}
                }

                // Folder
                if(mx>=600&&mx<760&&my>=306&&my<358&&!recording)chooseFolder(saveFolder,sizeof(saveFolder));

                // Piano
                int note=pianoHit(mx,my);
                if(note>=0){
                    int actual=std::clamp(note+octaveShift*12,0,127);
                    noteOn(actual);
                    g_keyDisplay[note]=true;
                    mouseBase=note; mouseActual=actual;
                }
                }
                break;

            case SDL_MOUSEBUTTONUP:
                for(auto&sl:sliders)sl.onUp();
                if(mouseBase>=0){
                    noteOff(mouseActual);
                    g_keyDisplay[mouseBase]=false;
                    mouseBase=-1; mouseActual=-1;
                }
                break;

            case SDL_MOUSEMOTION:
                for(auto&sl:sliders)sl.onMove(ev.motion.x);
                break;

            case SDL_KEYDOWN:
                if(!ev.key.repeat){
                    if(ev.key.keysym.sym==SDLK_F11){
                        fullscreen=!fullscreen;
                        SDL_SetWindowFullscreen(win,fullscreen?SDL_WINDOW_FULLSCREEN_DESKTOP:0);
                    }
                    if(ev.key.keysym.sym==SDLK_ESCAPE&&fullscreen){
                        fullscreen=false;
                        SDL_SetWindowFullscreen(win,0);
                    }
                    int base=keyToNote(ev.key.keysym.sym);
                    if(base>=0 && !keyHeld[base]){
                        int actual=std::clamp(base+octaveShift*12,0,127);
                        noteOn(actual);
                        keyHeld[base]=true;
                        keyActualNote[base]=actual;
                        g_keyDisplay[base]=true;
                    }
                }
                break;

            case SDL_KEYUP:
                {
                int base=keyToNote(ev.key.keysym.sym);
                if(base>=0 && keyHeld[base]){
                    noteOff(keyActualNote[base]);
                    keyHeld[base]=false;
                    g_keyDisplay[base]=false;
                }}
                break;
            }
        }

        // Sync atomic params
        g_vol.store(volVal);
        g_dist.store(distVal);
        g_wave.store(wave);
        g_dlyTime.store(dlyTime);
        g_dlyMix.store(dlyMix);
        g_reverb.store(rvbVal);
        g_lfoRate.store(lfoRateVal);
        g_lfoDepth.store(lfoDepthVal);
        g_lfoTgt.store(lfoTarget);
        g_lfoOn.store(lfoOn);

        // ── Render ──────────────────────────────────────────────────────────
        SDL_SetRenderDrawColor(ren,0,0,0,255); SDL_RenderClear(ren);
        fillR(0,0,WIN_W,WIN_H,{14,12,24});


        // Header
        fillR(0,0,WIN_W,44,{20,18,34}); wireR(0,0,WIN_W,44,{52,44,80});
        drawStrC(WIN_W/2,14,"AGRASYNTH",2,{218,196,255});

        // MIDI status
        { char mbuf[32]; std::snprintf(mbuf,sizeof(mbuf),"MIDI: %s",midiPortName.c_str());
          drawStr(8,16,mbuf,1,midiPortName=="NONE"?C4{80,72,110}:C4{80,200,120}); }

        // Recording status
        if(recording){
            int secs=g_recPos.load()/SR;
            char rbuf[20]; std::snprintf(rbuf,sizeof(rbuf),"REC %d:%02d",secs/60,secs%60);
            fillR(WIN_W-120,16,8,8,{220,40,40});
            drawStr(WIN_W-108,16,rbuf,1,{220,100,100});
        } else if(lastSaved[0]){
            drawStr(WIN_W-200,16,lastSaved,1,{60,180,90});
        }

        // Oscilloscope
        fillR(12,48,WIN_W-24,94,{12,10,22}); wireR(12,48,WIN_W-24,94,{52,44,80});
        drawOsc(18,52,WIN_W-36,86,distVal);

        // Controls panel background (covers waveform+sliders row and LFO row)
        fillR(12,146,WIN_W-24,154,{18,16,30}); wireR(12,146,WIN_W-24,154,{52,44,80});

        // Waveform buttons
        drawStrC(89,154,"WAVEFORM",1,{90,82,122});
        C4 bOn{88,50,170},bOff2{26,22,44},bdr{78,50,140};
        fillR(20, 170,130,42,wave==0?bOn:bOff2); wireR(20, 170,130,42,bdr);
        drawStrC(85, 184,"SINE",2,wave==0?C4{255,246,255}:C4{125,116,165});
        fillR(158,170,130,42,wave==1?bOn:bOff2); wireR(158,170,130,42,bdr);
        drawStrC(223,184,"SAW", 2,wave==1?C4{255,246,255}:C4{125,116,165});

        // Row 1 sliders (first 5)
        for(int i=0;i<5;i++)sliders[i].draw();

        // LFO section
        drawStrC(89,236,"LFO",1,{90,82,122});
        { C4 onBg=lfoOn?C4{88,50,170}:C4{26,22,44};
          fillR(20,248,140,38,onBg); wireR(20,248,140,38,bdr);
          drawStrC(90,261,lfoOn?"LFO ON":"LFO OFF",1,lfoOn?C4{255,246,255}:C4{125,116,165}); }
        sliders[5].draw(); // rate
        sliders[6].draw(); // depth

        // LFO target buttons
        fillR(680,248,120,38,lfoTarget==0?bOn:bOff2); wireR(680,248,120,38,bdr);
        drawStrC(740,261,"VIBRATO",1,lfoTarget==0?C4{255,246,255}:C4{125,116,165});
        fillR(812,248,120,38,lfoTarget==1?bOn:bOff2); wireR(812,248,120,38,bdr);
        drawStrC(872,261,"TREMOLO",1,lfoTarget==1?C4{255,246,255}:C4{125,116,165});

        // Separator
        fillR(0,302,WIN_W,1,{52,44,74});

        // Octave cluster
        drawStrC(109,308,"OCT",1,{90,82,122});
        char octBuf[6]; std::snprintf(octBuf,sizeof(octBuf),"%+d",octaveShift);
        drawStrC(109,319,octBuf,2,{200,182,255});
        fillR(22, 316,42,34,{32,28,52}); wireR(22, 316,42,34,{78,50,140});
        drawStrC(43, 325,"<",2,{180,160,230});
        fillR(152,316,42,34,{32,28,52}); wireR(152,316,42,34,{78,50,140});
        drawStrC(173,325,">",2,{180,160,230});

        // Play drone button
        fillR(230,306,166,52,playDrone?C4{148,28,62}:C4{26,128,60});
        wireR(230,306,166,52,{255,255,255,35});
        drawStrC(313,323,playDrone?"STOP":"PLAY",2,{255,255,255});

        // Record button
        C4 recBg=recording?C4{180,22,22}:C4{60,24,36};
        fillR(416,306,166,52,recBg); wireR(416,306,166,52,{255,255,255,35});
        drawStrC(499,323,recording?"STOP REC":"RECORD",2,{255,255,255});
        if(recording){
            int secs=g_recPos.load()/SR;
            char tbuf[12]; std::snprintf(tbuf,sizeof(tbuf),"%d:%02d",secs/60,secs%60);
            drawStrC(499,344,tbuf,1,{255,160,160});
        }

        // Folder button
        C4 fBg=(saveFolder[0])?C4{28,50,36}:C4{26,24,44};
        fillR(600,306,158,52,fBg); wireR(600,306,158,52,{255,255,255,35});
        drawStrC(679,316,"FOLDER",2,{200,220,200});
        if(saveFolder[0]){
            const char*last=strrchr(saveFolder,'/');
            const char*name=(last&&last[1])?last+1:(saveFolder[0]?saveFolder:"DESKTOP");
            char disp[16]={}; strncpy(disp,name,14);
            if(strlen(name)>14){disp[13]='.';disp[14]='\0';}
            drawStrC(679,336,disp,1,{120,180,130});
        } else {
            drawStrC(679,336,"DESKTOP",1,{100,120,100});
        }

        // Separator
        fillR(0,360,WIN_W,1,{52,44,74});

        // Octave strip
        fillR(0,364,WIN_W,26,{16,14,28});
        { char mkbuf[48];
          std::snprintf(mkbuf,sizeof(mkbuf),"OCT %+d   Z=C%d   Q=C%d   8 VOICES POLY",
              octaveShift,3+octaveShift,4+octaveShift);
          drawStrC(WIN_W/2,372,mkbuf,1,{80,72,108}); }

        drawPiano(g_keyDisplay);


        SDL_RenderPresent(ren);
    }

    delete midiIn;
    SDL_CloseAudioDevice(g_dev);
    delete[] g_recBuf;
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
