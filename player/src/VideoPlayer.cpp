#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "VideoPlayer.h"
#include "AudioOutput/AudioOutput.h"
#include "ChannelData/ChannelData.h"
#include "VideoSource/VideoSource.h"
#include "AudioSource/AudioSource.h"
#include "Displays/Display.h"
#include <list>
#include "JPEGDEC.h"

void VideoPlayer::_framePlayerTask(void *param)
{
  VideoPlayer *player = (VideoPlayer *)param;
  player->framePlayerTask();
}

void VideoPlayer::_audioPlayerTask(void *param)
{
  VideoPlayer *player = (VideoPlayer *)param;
  player->audioPlayerTask();
}

VideoPlayer::VideoPlayer(ChannelData *channelData, VideoSource *videoSource, AudioSource *audioSource, Display &display, AudioOutput *audioOutput)
: mChannelData(channelData), mVideoSource(videoSource), mAudioSource(audioSource), mDisplay(display), mState(VideoPlayerState::STOPPED), mAudioOutput(audioOutput)
{
}

void VideoPlayer::start()
{
  mVideoSource->start();
  if (mAudioSource != NULL)
  {
    mAudioSource->start();
  }
  // launch the frame player task
  xTaskCreatePinnedToCore(
      _framePlayerTask,
      "Frame Player",
      10000,
      this,
      1,
      NULL,
      0);
  if (mAudioSource != NULL && mAudioOutput != NULL)
  {
    xTaskCreatePinnedToCore(_audioPlayerTask, "audio_loop", 10000, this, 1, NULL, 0);
  }
}

void VideoPlayer::setChannel(int channel)
{
  mChannelData->setChannel(channel);
  // set the audio sample to 0 - TODO - move this somewhere else?
  mCurrentAudioSample = 0;
  mChannelVisible = millis();
  // update the video source
  mVideoSource->setChannel(channel);
}

void VideoPlayer::play()
{
  if (mState == VideoPlayerState::PLAYING)
  {
    return;
  }
  mState = VideoPlayerState::PLAYING;
  mVideoSource->setState(VideoPlayerState::PLAYING);
  mCurrentAudioSample = 0;
}

void VideoPlayer::stop()
{
  if (mState == VideoPlayerState::STOPPED)
  {
    return;
  }
  mState = VideoPlayerState::STOPPED;
  mVideoSource->setState(VideoPlayerState::STOPPED);
  mCurrentAudioSample = 0;
  mDisplay.fillScreen(DisplayColors::BLACK);
}

void VideoPlayer::pause()
{
  if (mState == VideoPlayerState::PAUSED)
  {
    return;
  }
  mState = VideoPlayerState::PAUSED;
  mVideoSource->setState(VideoPlayerState::PAUSED);
}

void VideoPlayer::playStatic()
{
  if (mState == VideoPlayerState::STATIC)
  {
    return;
  }
  mState = VideoPlayerState::STATIC;
  mVideoSource->setState(VideoPlayerState::STATIC);
}


// double buffer the dma drawing otherwise we get corruption
uint16_t *dmaBuffer[2] = {NULL, NULL};
int dmaBufferIndex = 0;
int _doDraw(JPEGDRAW *pDraw)
{
  if (pDraw == NULL || pDraw->pPixels == NULL || pDraw->iWidth <= 0 || pDraw->iHeight <= 0)
  {
    return 0;
  }
  VideoPlayer *player = (VideoPlayer *)pDraw->pUser;
  player->mDisplay.drawPixels(pDraw->x, pDraw->y, pDraw->iWidth, pDraw->iHeight, pDraw->pPixels);
  return 1;
}

static unsigned short x = 12345, y = 6789, z = 42, w = 1729;

unsigned short xorshift16()
{
  unsigned short t = x ^ (x << 5);
  x = y;
  y = z;
  z = w;
  w = w ^ (w >> 1) ^ t ^ (t >> 3);
  return w & 0xFFFF;
}

void VideoPlayer::framePlayerTask()
{
  uint16_t *staticBuffer = NULL;
  uint8_t *jpegBuffer = NULL;
  size_t jpegBufferLength = 0;
  size_t jpegLength = 0;
  unsigned long drawCount = 0;
  unsigned long lastDrawLogMs = 0;
  // used for calculating frame rate
  std::list<int> frameTimes;
  while (true)
  {
    if (mState == VideoPlayerState::STOPPED || mState == VideoPlayerState::PAUSED)
    {
      // nothing to do - just wait
      vTaskDelay(100 / portTICK_PERIOD_MS);
      continue;
    }
    if (mState == VideoPlayerState::STATIC)
    {
      // draw random pixels to the screen to simulate static
      // we'll do this 8 rows of pixels at a time to save RAM
      int width = mDisplay.width();
      int height = 8;
      if (staticBuffer == NULL)
      {
        staticBuffer = (uint16_t *)malloc(width * height * 2);
      }
      mDisplay.startWrite();
      for (int i = 0; i < mDisplay.height(); i++)
      {
        for (int p = 0; p < width * height; p++)
        {
          int grey = xorshift16() >> 8;
          staticBuffer[p] = mDisplay.color565(grey, grey, grey);
        }
        mDisplay.drawPixels(0, i * height, width, height, staticBuffer);
      }
      mDisplay.endWrite();
      vTaskDelay(50 / portTICK_PERIOD_MS);
      continue;
    }
    // get the next frame
    if (!mVideoSource->getVideoFrame(&jpegBuffer, jpegBufferLength, jpegLength))
    {
      // no frame ready yet
      vTaskDelay(10 / portTICK_PERIOD_MS);
      continue;
    }
    frameTimes.push_back(millis());
    // keep the frame rate elapsed time to 5 seconds
    while(frameTimes.size() > 0 && frameTimes.back() - frameTimes.front() > 5000) {
      frameTimes.pop_front();
    }
    mDisplay.startWrite();
    if (mJpeg.openRAM(jpegBuffer, jpegLength, _doDraw))
    {
      mJpeg.setUserPointer(this);
      mJpeg.setPixelType(RGB565_LITTLE_ENDIAN);
      int decodeOptions = 0;
      int drawX = 0;
      int drawY = 0;
      int srcW = mJpeg.getWidth();
      int srcH = mJpeg.getHeight();
      int dstW = mDisplay.width();
      int dstH = mDisplay.height();
      if (srcW > dstW || srcH > dstH)
      {
        if ((srcW / 2) <= dstW && (srcH / 2) <= dstH)
        {
          decodeOptions = JPEG_SCALE_HALF;
        }
        else if ((srcW / 4) <= dstW && (srcH / 4) <= dstH)
        {
          decodeOptions = JPEG_SCALE_QUARTER;
        }
        else
        {
          decodeOptions = JPEG_SCALE_EIGHTH;
        }
      }
      int outW = srcW;
      int outH = srcH;
      if (decodeOptions == JPEG_SCALE_HALF)
      {
        outW = srcW / 2;
        outH = srcH / 2;
      }
      else if (decodeOptions == JPEG_SCALE_QUARTER)
      {
        outW = srcW / 4;
        outH = srcH / 4;
      }
      else if (decodeOptions == JPEG_SCALE_EIGHTH)
      {
        outW = srcW / 8;
        outH = srcH / 8;
      }
      if (outW > 0 && outH > 0)
      {
        drawX = (dstW - outW) / 2;
        drawY = (dstH - outH) / 2;
        if (drawX < 0) drawX = 0;
        if (drawY < 0) drawY = 0;
      }
      mJpeg.decode(drawX, drawY, decodeOptions);
      mJpeg.close();
      drawCount++;
      if (millis() - lastDrawLogMs > 2000) {
        Serial.printf(
          "Frame Drawn: count=%lu len=%u src=%dx%d dst=%dx%d opt=%d pos=%d,%d\n",
          drawCount,
          (unsigned int)jpegLength,
          srcW,
          srcH,
          dstW,
          dstH,
          decodeOptions,
          drawX,
          drawY
        );
        lastDrawLogMs = millis();
      }
    }
    // show channel indicator 
    if (millis() - mChannelVisible < 2000) {
      mDisplay.drawChannel(mChannelData->getChannelNumber());
    }
    #if CORE_DEBUG_LEVEL > 0
    mDisplay.drawFPS(frameTimes.size() / 5);
    #endif
    mDisplay.endWrite();
    // Prevent starving Wi-Fi/system tasks on ESP32-S3 under continuous decode.
    vTaskDelay(1 / portTICK_PERIOD_MS);
  }
}

void VideoPlayer::audioPlayerTask()
{
  if (mAudioSource == NULL || mAudioOutput == NULL)
  {
    vTaskDelete(NULL);
    return;
  }
  size_t bufferLength = 16000;
  uint8_t *audioData = (uint8_t *)malloc(16000);
  while (true)
  {
    if (mState != VideoPlayerState::PLAYING)
    {
      // nothing to do - just wait
      vTaskDelay(100 / portTICK_PERIOD_MS);
      continue;
    }
    // get audio data to play
    int audioLength = mAudioSource->getAudioSamples(&audioData, bufferLength, mCurrentAudioSample);
    // have we reached the end of the channel?
    if (audioLength == 0) {
      // we want to loop the video so reset the channel data and start again
      stop();
      setChannel(mChannelData->getChannelNumber());
      play();
      continue;
    }
    if (audioLength > 0) {
      // play the audio
      for(int i=0; i<audioLength; i+=1000) {
        mAudioOutput->write(audioData + i, min(1000, audioLength - i));
        mCurrentAudioSample += min(1000, audioLength - i);
        if (mState != VideoPlayerState::PLAYING)
        {
          mCurrentAudioSample = 0;
          mVideoSource->updateAudioTime(0);
          break;
        }
        mVideoSource->updateAudioTime(1000 * mCurrentAudioSample / 16000);
      }
    }
    else
    {
      vTaskDelay(100 / portTICK_PERIOD_MS);
    }
  }
}
