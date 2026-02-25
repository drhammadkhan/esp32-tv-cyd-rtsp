#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Stream.h>
#include "NetworkVideoSource.h"
#include "../ChannelData/NetworkChannelData.h"

void NetworkVideoSource::_frameDownloaderTask(void *param)
{
  NetworkVideoSource *networkVideoSource = (NetworkVideoSource *)param;
  networkVideoSource->frameDownloaderTask();
}

void NetworkVideoSource::frameDownloaderTask()
{
  HTTPClient http;
  http.setReuse(true);
  uint8_t *downloadBuffer = NULL;
  int downloadBufferLength = 0;
  while (true)
  {
    if (mState == VideoPlayerState::STOPPED || mState == VideoPlayerState::STATIC)
    {
      vTaskDelay(100 / portTICK_PERIOD_MS);
      continue;
    }
    if (mState == VideoPlayerState::PAUSED)
    {
      // video time is not passing, so keep moving the start time forward so it is now
      mLastAudioTimeUpdateMs = millis();
      vTaskDelay(100 / portTICK_PERIOD_MS);
      continue;
    }
    // do we need to download a frame?
    if (mFrameReady)
    {
      // we already have a frame ready, so just wait
      vTaskDelay(10 / portTICK_PERIOD_MS);
      continue;
    }
    // work out the video time from a combination of the currentAudioSample and the elapsed time
    int elapsedTime = millis() - mLastAudioTimeUpdateMs;
    int videoTime = mAudioTimeMs + elapsedTime;
    if (WiFi.status() == WL_CONNECTED)
    {
      std::string url = mChannelData->getFrameURL() + "/" + std::to_string(videoTime);
      http.begin(url.c_str());
int httpCode = http.GET();

      if (httpCode == HTTP_CODE_OK)
      {
        // read the image into our local buffer
        int jpegLength = http.getSize();
        if (jpegLength > 0) {
          if (jpegLength > downloadBufferLength)
          {
            uint8_t *newDownloadBuffer = (uint8_t *)realloc(downloadBuffer, jpegLength);
            if (newDownloadBuffer == NULL)
            {
              Serial.printf("OOM allocating download buffer: %d bytes\n", jpegLength);
              vTaskDelay(10 / portTICK_PERIOD_MS);
              http.end();
              continue;
            }
            downloadBuffer = newDownloadBuffer;
            downloadBufferLength = jpegLength;
          }
          int bytesRead = http.getStreamPtr()->readBytes(downloadBuffer, jpegLength);
          if (bytesRead <= 0)
          {
            vTaskDelay(10 / portTICK_PERIOD_MS);
            http.end();
            continue;
          }
          jpegLength = bytesRead;
        } else {
          // Handle chunked/unknown-length responses by reading all available bytes.
          Stream *stream = http.getStreamPtr();
          int total = 0;
          unsigned long lastRead = millis();
          while (stream && (stream->available() > 0 || millis() - lastRead < 200))
          {
            int available = stream->available();
            if (available <= 0)
            {
              vTaskDelay(1 / portTICK_PERIOD_MS);
              continue;
            }
            if (total + available > downloadBufferLength)
            {
              size_t needed = total + available + 1024;
              uint8_t *newDownloadBuffer = (uint8_t *)realloc(downloadBuffer, needed);
              if (newDownloadBuffer == NULL)
              {
                Serial.printf("OOM growing download buffer: %u bytes\n", (unsigned int)needed);
                break;
              }
              downloadBuffer = newDownloadBuffer;
              downloadBufferLength = needed;
            }
            int read = stream->readBytes(downloadBuffer + total, available);
            if (read > 0)
            {
              total += read;
              lastRead = millis();
            }
          }
          jpegLength = total;
        }
        if (jpegLength <= 0) {
          vTaskDelay(10 / portTICK_PERIOD_MS);
          continue;
        }
        // lock the image buffer
        xSemaphoreTake(mCurrentFrameMutex, portMAX_DELAY);
        // reallocate the image buffer if necessary
        if (jpegLength > mCurrentFrameBufferLength)
        {
          uint8_t *newFrameBuffer = (uint8_t *)realloc(mCurrentFrameBuffer, jpegLength);
          if (newFrameBuffer == NULL)
          {
            Serial.printf("OOM allocating frame buffer: %d bytes\n", jpegLength);
            xSemaphoreGive(mCurrentFrameMutex);
            http.end();
            vTaskDelay(10 / portTICK_PERIOD_MS);
            continue;
          }
          mCurrentFrameBuffer = newFrameBuffer;
          mCurrentFrameBufferLength = jpegLength;
        }
        // copy the image buffer
        memcpy(mCurrentFrameBuffer, downloadBuffer, jpegLength);
        mCurrentFrameLength = jpegLength;
        // don't set this flag if we aren't playing otherwise we might trigger a draw
        if (mState == VideoPlayerState::PLAYING)
        {
          mFrameReady = true;
        }
        // unlock the image buffer
        xSemaphoreGive(mCurrentFrameMutex);
        // Serial.printf("Read %d bytes in %d ms\n", download_image_length, millis() - start_download_time);
      }
      else
      {
        Serial.printf("HTTP error: %d\n", httpCode);
        vTaskDelay(10 / portTICK_PERIOD_MS);
      }
    }
    else
    {
      Serial.println("Not connected to WiFi");
      delay(1000);
    }
  }
}


NetworkVideoSource::NetworkVideoSource(NetworkChannelData *channelData) : mChannelData(channelData)
{
}

void NetworkVideoSource::start() {
  // create a mutex to control access to the JPEG buffer
  mCurrentFrameMutex = xSemaphoreCreateMutex();
  // launch the frame downloader task
  xTaskCreatePinnedToCore(
      _frameDownloaderTask,
      "Frame Downloader",
      10000,
      this,
      1,
      NULL,
      0);
}

bool NetworkVideoSource::getVideoFrame(uint8_t **buffer, size_t &bufferLength, size_t &frameLength) {
  if(mCurrentFrameBuffer == NULL) {
    return false;
  }
  bool copiedFrame = false;
  // lock the image buffer
  xSemaphoreTake(mCurrentFrameMutex, portMAX_DELAY);
  // if the frame is ready, copy it to the buffer
  if (mFrameReady) {
    mFrameReady=false;
    copiedFrame = true;
    // reallocate the image buffer if necessary
    if (mCurrentFrameBufferLength > bufferLength) {
      uint8_t *newBuffer = (uint8_t *)realloc(*buffer, mCurrentFrameBufferLength);
      if (newBuffer == NULL) {
        copiedFrame = false;
        xSemaphoreGive(mCurrentFrameMutex);
        return false;
      }
      *buffer = newBuffer;
      bufferLength = mCurrentFrameBufferLength;
    }
    // copy the image buffer
    memcpy(*buffer, mCurrentFrameBuffer, mCurrentFrameLength);
    frameLength = mCurrentFrameLength;
  }
  // unlock the image buffer
  xSemaphoreGive(mCurrentFrameMutex);
  // return true if we copied a frame, false otherwise
  return copiedFrame;
}

void NetworkVideoSource::setChannel(int channel) {
  mLastAudioTimeUpdateMs = millis();
  mFrameReady = false;
  mAudioTimeMs = 0;
}
