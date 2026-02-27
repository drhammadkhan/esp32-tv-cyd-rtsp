#ifndef LED_MATRIX
#include <Arduino.h>
#include <TFT_eSPI.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "TFT.h"

TFT::TFT(): tft(new TFT_eSPI()) {
  mDisplayMutex = xSemaphoreCreateRecursiveMutex();
}

void TFT::ensureInit() {
  if (mInitialized) {
    return;
  }
  if (mDisplayMutex != NULL) {
    xSemaphoreTakeRecursive(mDisplayMutex, portMAX_DELAY);
    if (mInitialized) {
      xSemaphoreGiveRecursive(mDisplayMutex);
      return;
    }
  }
  // power on the tft
  #ifdef TFT_POWER
  if (TFT_POWER != GPIO_NUM_NC) {
    Serial.println("Powering on TFT");
    pinMode(TFT_POWER, OUTPUT);
    digitalWrite(TFT_POWER, TFT_POWER_ON);
  }
  #endif

  tft->init();
  #ifdef ES3N28P_ILI9341V
  #ifndef ES3N28P_ROTATION
  #define ES3N28P_ROTATION 1
  #endif
  tft->setRotation(ES3N28P_ROTATION);
  #elif defined(M5CORE2)
  tft->setRotation(6);
  #else
  tft->setRotation(1);
  #endif
  Serial.printf("TFT init done: w=%d h=%d rot=%d\n", tft->width(), tft->height(),
  #ifdef ES3N28P_ILI9341V
    ES3N28P_ROTATION
  #elif defined(M5CORE2)
    6
  #else
    1
  #endif
  );
  tft->fillScreen(TFT_BLACK);
  #ifdef USE_DMA
  tft->initDMA();
  #endif
  tft->setSwapBytes(true);
  #ifndef TFT_INVERT_DISPLAY
  #define TFT_INVERT_DISPLAY 1
  #endif
  tft->invertDisplay(TFT_INVERT_DISPLAY);
  tft->fillScreen(TFT_BLACK);
  tft->setTextFont(2);
  tft->setTextSize(2);
  tft->setTextColor(TFT_GREEN, TFT_BLACK);
  mInitialized = true;
  if (mDisplayMutex != NULL) {
    xSemaphoreGiveRecursive(mDisplayMutex);
  }
}

void TFT::drawPixels(int x, int y, int width, int height, uint16_t *pixels) {
  ensureInit();
#ifdef ES3N28P_ILI9341V
  // On this panel, pushImage is more reliable than raw pushPixels window writes.
  tft->pushImage(x, y, width, height, pixels);
  return;
#endif
  int numPixels = width * height;
  if (dmaBuffer[dmaBufferIndex] == NULL)
  {
    dmaBuffer[dmaBufferIndex] = (uint16_t *)malloc(numPixels * 2);
  }
  memcpy(dmaBuffer[dmaBufferIndex], pixels, numPixels * 2);
  #ifdef USE_DMA
  tft->dmaWait();
  #endif
  tft->setAddrWindow(x, y, width, height);
  #ifdef USE_DMA
  tft->pushPixelsDMA(dmaBuffer[dmaBufferIndex], numPixels);
  #else
  tft->pushPixels(dmaBuffer[dmaBufferIndex], numPixels);
  #endif
  dmaBufferIndex = (dmaBufferIndex + 1) % 2;
}

void TFT::startWrite() {
  ensureInit();
  if (mDisplayMutex != NULL) {
    xSemaphoreTakeRecursive(mDisplayMutex, portMAX_DELAY);
  }
  tft->startWrite();
}

void TFT::endWrite() {
  tft->endWrite();
  if (mDisplayMutex != NULL) {
    xSemaphoreGiveRecursive(mDisplayMutex);
  }
}

int TFT::width() {
  ensureInit();
  return tft->width();
}

int TFT::height() {
  ensureInit();
  return tft->height();
}

void TFT::fillScreen(uint16_t color) {
  ensureInit();
  if (mDisplayMutex != NULL) {
    xSemaphoreTakeRecursive(mDisplayMutex, portMAX_DELAY);
  }
  tft->fillScreen(color);
  if (mDisplayMutex != NULL) {
    xSemaphoreGiveRecursive(mDisplayMutex);
  }
}

void TFT::drawChannel(int channelIndex) {
  ensureInit();
  if (mDisplayMutex != NULL) {
    xSemaphoreTakeRecursive(mDisplayMutex, portMAX_DELAY);
  }
  tft->setCursor(20, 20);
  tft->setTextColor(TFT_GREEN, TFT_BLACK);
  tft->printf("%d", channelIndex);
  if (mDisplayMutex != NULL) {
    xSemaphoreGiveRecursive(mDisplayMutex);
  }
}

void TFT::drawTuningText(const char *serverInfo) {
  ensureInit();
  if (mDisplayMutex != NULL) {
    xSemaphoreTakeRecursive(mDisplayMutex, portMAX_DELAY);
  }
  tft->fillScreen(TFT_BLACK);
  tft->setTextSize(2);
  tft->setCursor(20, 20);
  tft->setTextColor(TFT_GREEN, TFT_BLACK);
  tft->println("TUNING...");
  if (serverInfo != nullptr && serverInfo[0] != '\0') {
    tft->setTextSize(1);
    tft->setCursor(20, 56);
    tft->setTextColor(TFT_WHITE, TFT_BLACK);
    tft->print("Server: ");
    tft->println(serverInfo);
    tft->setTextSize(2);
  }
  if (mDisplayMutex != NULL) {
    xSemaphoreGiveRecursive(mDisplayMutex);
  }
}

void TFT::drawSDCardFailed() {
  ensureInit();
  if (mDisplayMutex != NULL) {
    xSemaphoreTakeRecursive(mDisplayMutex, portMAX_DELAY);
  }
  tft->fillScreen(TFT_RED);
  tft->setCursor(0, 20);
  tft->setTextColor(TFT_WHITE);
  tft->setTextSize(2);
  tft->println("Failed to mount SD Card");
  if (mDisplayMutex != NULL) {
    xSemaphoreGiveRecursive(mDisplayMutex);
  }
}

void TFT::drawFPS(int fps) {
    ensureInit();
    if (mDisplayMutex != NULL) {
      xSemaphoreTakeRecursive(mDisplayMutex, portMAX_DELAY);
    }
    // show the frame rate in the top right
    tft->setCursor(width() - 50, 20);
    tft->setTextColor(TFT_GREEN, TFT_BLACK);
    tft->printf("%d", fps);
    if (mDisplayMutex != NULL) {
      xSemaphoreGiveRecursive(mDisplayMutex);
    }
}

bool TFT::hasTouch() {
  ensureInit();
#if defined(TOUCH_CS) && (TOUCH_CS != -1)
  return true;
#else
  return false;
#endif
}

bool TFT::getTouchPoint(uint16_t *x, uint16_t *y) {
  ensureInit();
#if defined(TOUCH_CS) && (TOUCH_CS != -1)
  if (mDisplayMutex == NULL) {
    return tft->getTouch(x, y);
  }
  if (xSemaphoreTakeRecursive(mDisplayMutex, 0) != pdTRUE) {
    return false;
  }
  bool pressed = tft->getTouch(x, y);
  xSemaphoreGiveRecursive(mDisplayMutex);
  return pressed;
#else
  (void)x;
  (void)y;
  return false;
#endif
}
#endif
