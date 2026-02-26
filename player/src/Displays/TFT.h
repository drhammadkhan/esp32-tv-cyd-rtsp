#pragma once
#ifndef LED_MATRIX
#include "Display.h"

class TFT_eSPI;
struct QueueDefinition;
typedef struct QueueDefinition * QueueHandle_t;
typedef QueueHandle_t SemaphoreHandle_t;

class TFT: public Display {
private:
  TFT_eSPI *tft;
  uint16_t *dmaBuffer[2] = {NULL, NULL};
  int dmaBufferIndex = 0;
  SemaphoreHandle_t mDisplayMutex = NULL;
public:
  TFT();
  void drawPixels(int x, int y, int width, int height, uint16_t *pixels);
  void startWrite();
  void endWrite();
  int width();
  int height();
  void fillScreen(uint16_t color);
  void drawChannel(int channelIndex);
  void drawTuningText(const char *serverInfo = nullptr);
  void drawFPS(int fps);
  void drawSDCardFailed();
  bool hasTouch();
  bool getTouchPoint(uint16_t *x, uint16_t *y);
};
#endif
