#ifndef LED_MATRIX
#include <Arduino.h>
#include <TFT_eSPI.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "TFT.h"

#ifdef ES3N28P_ILI9341V
static void applyES3N28PInit(TFT_eSPI *tft) {
  auto w = [tft](uint8_t reg, std::initializer_list<uint8_t> data) {
    tft->writecommand(reg);
    for (uint8_t d : data) {
      tft->writedata(d);
    }
  };
  w(0xCF, {0x00, 0xC1, 0x30});
  w(0xED, {0x64, 0x03, 0x12, 0x81});
  w(0xE8, {0x85, 0x00, 0x78});
  w(0xCB, {0x39, 0x2C, 0x00, 0x34, 0x02});
  w(0xF7, {0x20});
  w(0xEA, {0x00, 0x00});
  w(0xC0, {0x13});
  w(0xC1, {0x13});
  w(0xC5, {0x22, 0x35});
  w(0xC7, {0xBD});
  w(0x21, {});
  w(0x36, {0x08});
  w(0xB6, {0x0A, 0xA2});
  w(0x3A, {0x55});
  w(0xF6, {0x01, 0x30});
  w(0xB1, {0x00, 0x1B});
  w(0xF2, {0x00});
  w(0x26, {0x01});
  w(0xE0, {0x0F, 0x35, 0x31, 0x0B, 0x0E, 0x06, 0x49, 0xA7, 0x33, 0x07, 0x0F, 0x03, 0x0C, 0x0A, 0x00});
  w(0xE1, {0x00, 0x0A, 0x0F, 0x04, 0x11, 0x08, 0x36, 0x58, 0x4D, 0x07, 0x10, 0x0C, 0x32, 0x34, 0x0F});
  w(0x11, {});
  delay(120);
  w(0x29, {});
}
#endif

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
  applyES3N28PInit(tft);
#endif
  #ifdef ES3N28P_ILI9341V
  tft->setRotation(1);
  // ES3N28P panel needs a custom MADCTL in landscape to avoid vertical mirroring.
  tft->writecommand(0x36);
  tft->writedata(0x68); // BGR | MX | MV (180 deg from previous mapping)
  #elif defined(M5CORE2)
  tft->setRotation(6);
  #else
  tft->setRotation(1);
  #endif
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
