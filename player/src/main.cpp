#include <Arduino.h>
#include <WiFi.h>
#include "Displays/TFT.h"
#include "Displays/Matrix.h"
#include "RemoteInput.h"
#include "VideoPlayer.h"
#include "AudioOutput/I2SOutput.h"
#include "AudioOutput/DACOutput.h"
#include "AudioOutput/PDMTimerOutput.h"
#include "AudioOutput/PWMTimerOutput.h"
#include "AudioOutput/PDMOutput.h"
#include "ChannelData/NetworkChannelData.h"
#include "ChannelData/SDCardChannelData.h"
#include "AudioSource/NetworkAudioSource.h"
#include "VideoSource/NetworkVideoSource.h"
#include "AudioSource/SDCardAudioSource.h"
#include "VideoSource/SDCardVideoSource.h"
#include "AVIParser/AVIParser.h"
#include "SDCard.h"
#include "PowerUtils.h"
#include "Button.h"
#if !defined(IGNORE_LOCAL_OVERRIDES) && __has_include("LocalOverrides.h")
#include "LocalOverrides.h"
#endif

#ifndef WIFI_SSID
#define WIFI_SSID "YOUR_WIFI_SSID"
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
#endif

#ifndef VIDEO_SERVER_HOST
#define VIDEO_SERVER_HOST "192.168.1.100"
#endif

#ifndef VIDEO_SERVER_PORT
#define VIDEO_SERVER_PORT 8123
#endif

#define STRINGIFY_INNER(x) #x
#define STRINGIFY(x) STRINGIFY_INNER(x)

#ifndef VIDEO_SERVER_PORT_STR
#define VIDEO_SERVER_PORT_STR STRINGIFY(VIDEO_SERVER_PORT)
#endif

#ifndef FRAME_ENDPOINT
#define FRAME_ENDPOINT "frame"
#endif

#ifdef DISABLE_AUDIO
#warning "Audio disabled for higher frame rate"
#endif

char FRAME_URL[128] = {0};
char AUDIO_URL[128] = {0};
char CHANNEL_INFO_URL[128] = {0};
char CLIENT_QUERY[48] = {0};
char TUNING_SERVER_INFO[80] = {0};

#ifdef HAS_IR_REMOTE
RemoteInput *remoteInput = NULL;
#else
#ifndef HAS_BUTTONS
#warning "No Remote Input - Will default to playing channel 0"
#endif
#endif

#ifndef USE_DMA
#warning "No DMA - Drawing may be slower"
#endif

VideoSource *videoSource = NULL;
AudioSource *audioSource = NULL;
VideoPlayer *videoPlayer = NULL;
AudioOutput *audioOutput = NULL;
ChannelData *channelData = NULL;
#ifdef LED_MATRIX
Matrix display;
#else
TFT display;
#endif

bool touchTracking = false;
int16_t touchStartX = 0;
int16_t touchStartY = 0;
int16_t touchEndX = 0;
int16_t touchEndY = 0;
unsigned long lastSwipeMillis = 0;
unsigned long lastTouchPollMillis = 0;
bool lastChannelButtonPressed = false;
unsigned long lastChannelButtonMillis = 0;
bool channelButtonRawState = false;
unsigned long channelButtonRawChangedMillis = 0;
bool channelButtonLatched = false;
unsigned long channelButtonReleaseStableMillis = 0;

void setup()
{
  Serial.begin(115200);
  Serial.printf("Total heap: %d\n", ESP.getHeapSize());
  Serial.printf("Free heap: %d\n", ESP.getFreeHeap());
  Serial.printf("Total PSRAM: %d\n", ESP.getPsramSize());
  Serial.printf("Free PSRAM: %d\n", ESP.getFreePsram());
  powerInit();
  buttonInit();
  if (display.hasTouch()) {
    Serial.println("Touch input enabled");
  }
#ifdef CHANNEL_SWITCH_BUTTON
  pinMode(CHANNEL_SWITCH_BUTTON, INPUT_PULLUP);
#endif
  #ifdef USE_SDCARD
  Serial.println("Using SD Card");
  // power on the SD card
  #ifdef SD_CARD_PWR
  if (SD_CARD_PWR != GPIO_NUM_NC) {
    pinMode(SD_CARD_PWR, OUTPUT);
    digitalWrite(SD_CARD_PWR, SD_CARD_PWR_ON);
  }
  #endif
  #ifdef USE_SDIO
  SDCard *card = new SDCard(SD_CARD_CLK, SD_CARD_CMD, SD_CARD_D0, SD_CARD_D1, SD_CARD_D2, SD_CARD_D3);
  #else
  SDCard *card = new SDCard(SD_CARD_MISO, SD_CARD_MOSI, SD_CARD_CLK, SD_CARD_CS);
  #endif
  // check that the SD Card has mounted properly
  if (!card->isMounted()) {
    Serial.println("Failed to mount SD Card");
    display.drawSDCardFailed();
    while(true) {
      delay(1000);
    }
  }
  channelData = new SDCardChannelData(card, "/");
  audioSource = new SDCardAudioSource((SDCardChannelData *) channelData);
  videoSource = new SDCardVideoSource((SDCardChannelData *) channelData);
  #else
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  WiFi.setSleep(false);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);
  Serial.println("");
  // disable WiFi power saving for speed
  Serial.println("WiFi connected");
  int videoServerPort = atoi(VIDEO_SERVER_PORT_STR);
  if (videoServerPort <= 0 || videoServerPort > 65535) {
    videoServerPort = VIDEO_SERVER_PORT;
  }
  snprintf(TUNING_SERVER_INFO, sizeof(TUNING_SERVER_INFO), "%s:%d", VIDEO_SERVER_HOST, videoServerPort);
  snprintf(FRAME_URL, sizeof(FRAME_URL), "http://%s:%d/%s", VIDEO_SERVER_HOST, videoServerPort, FRAME_ENDPOINT);
  snprintf(AUDIO_URL, sizeof(AUDIO_URL), "http://%s:%d/audio", VIDEO_SERVER_HOST, videoServerPort);
  snprintf(CHANNEL_INFO_URL, sizeof(CHANNEL_INFO_URL), "http://%s:%d/channel_info", VIDEO_SERVER_HOST, videoServerPort);
  uint64_t chipId = ESP.getEfuseMac();
  snprintf(CLIENT_QUERY, sizeof(CLIENT_QUERY), "?cid=esp32-%04X%08X", (uint16_t)(chipId >> 32), (uint32_t)chipId);
  Serial.printf("Using video server: %s:%d\n", VIDEO_SERVER_HOST, videoServerPort);
  channelData = new NetworkChannelData(CHANNEL_INFO_URL, FRAME_URL, AUDIO_URL, CLIENT_QUERY);
  videoSource = new NetworkVideoSource((NetworkChannelData *) channelData);
#ifndef DISABLE_AUDIO
  audioSource = new NetworkAudioSource((NetworkChannelData *) channelData);
#endif
  #endif

#ifdef HAS_IR_REMOTE
  remoteInput = new RemoteInput(IR_RECV_PIN, IR_RECV_PWR, IR_RECV_GND, IR_RECV_IND);
  remoteInput->start();
#endif

#ifdef USE_DAC_AUDIO
#ifndef DISABLE_AUDIO
  audioOutput = new DACOutput(I2S_NUM_0);
  audioOutput->start(16000);
#endif
#endif
#ifdef PDM_GPIO_NUM
#ifndef DISABLE_AUDIO
  i2s speaker pins
  i2s_pin_config_t i2s_speaker_pins = {
      .bck_io_num = I2S_PIN_NO_CHANGE,
      .ws_io_num = GPIO_NUM_0,
      .data_out_num = PDM_GPIO_NUM,
      .data_in_num = I2S_PIN_NO_CHANGE};
  audioOutput = new PDMOutput(I2S_NUM_0, i2s_speaker_pins);
  audioOutput->start(16000);
#endif
#endif
#ifdef PWM_GPIO_NUM
#ifndef DISABLE_AUDIO
  audioOutput = new PWMTimerOutput(PWM_GPIO_NUM);
  audioOutput->start(16000);
#endif
#endif
#ifdef I2S_SPEAKER_SERIAL_CLOCK
#ifndef DISABLE_AUDIO
#ifdef SPK_MODE
  pinMode(SPK_MODE, OUTPUT);
  digitalWrite(SPK_MODE, HIGH);
#endif
  // i2s speaker pins
  i2s_pin_config_t i2s_speaker_pins = {
      .bck_io_num = I2S_SPEAKER_SERIAL_CLOCK,
      .ws_io_num = I2S_SPEAKER_LEFT_RIGHT_CLOCK,
      .data_out_num = I2S_SPEAKER_SERIAL_DATA,
      .data_in_num = I2S_PIN_NO_CHANGE};

  audioOutput = new I2SOutput(I2S_NUM_1, i2s_speaker_pins);
  audioOutput->start(16000);
#endif
#endif
  videoPlayer = new VideoPlayer(
    channelData,
    videoSource,
    audioSource,
    display,
    audioOutput
  );
  videoPlayer->start();
#ifndef HAS_IR_REMOTE
  display.drawTuningText(TUNING_SERVER_INFO);
  // get the channel info
  while(!channelData->fetchChannelData()) {
    Serial.println("Failed to fetch channel data");
    delay(1000);
  }
  // default to first channel
  videoPlayer->setChannel(0);
  delay(500);
  videoPlayer->play();
#endif
  #ifdef M5CORE2
  if (audioOutput != NULL) {
    audioOutput->setVolume(4);
  }
  #endif
}

int channel = 0;

void volumeUp() {
  if (audioOutput == NULL) return;
  audioOutput->volumeUp();
  delay(500);
  Serial.println("VOLUME_UP");
}

void volumeDown() {
  if (audioOutput == NULL) return;
  audioOutput->volumeDown();
  delay(500);
  Serial.println("VOLUME_DOWN");
}

void channelDown() {
  videoPlayer->playStatic();
  delay(500);
  channel--;
  if (channel < 0) {
    channel = channelData->getChannelCount() - 1;
  }
  videoPlayer->setChannel(channel);
  videoPlayer->play();
  Serial.printf("CHANNEL_DOWN %d\n", channel);
}

void channelUp() {
  videoPlayer->playStatic();
  delay(500);
  channel = (channel + 1) % channelData->getChannelCount();
  videoPlayer->setChannel(channel);
  videoPlayer->play();
  Serial.printf("CHANNEL_UP %d\n", channel);
}

void handleTouchSwipe() {
  if (!display.hasTouch()) {
    return;
  }
  if (millis() - lastTouchPollMillis < 40) {
    return;
  }
  lastTouchPollMillis = millis();
  uint16_t x = 0;
  uint16_t y = 0;
  bool pressed = display.getTouchPoint(&x, &y);
  if (pressed) {
    if (!touchTracking) {
      touchTracking = true;
      touchStartX = x;
      touchStartY = y;
      touchEndX = x;
      touchEndY = y;
    } else {
      touchEndX = x;
      touchEndY = y;
    }
    return;
  }
  if (!touchTracking) {
    return;
  }
  touchTracking = false;
  int dx = touchEndX - touchStartX;
  int dy = touchEndY - touchStartY;
  int absDx = abs(dx);
  int absDy = abs(dy);
  if (absDx < 120 || absDx < (absDy + 30)) {
    return;
  }
  if (millis() - lastSwipeMillis < 500) {
    return;
  }
  lastSwipeMillis = millis();
  if (dx < 0) {
    Serial.println("SWIPE_RIGHT_TO_LEFT -> CHANNEL_UP");
    channelUp();
  }
}

void handleChannelSwitchButton() {
#ifdef CHANNEL_SWITCH_BUTTON
  unsigned long now = millis();
  bool rawPressed = (digitalRead(CHANNEL_SWITCH_BUTTON) == LOW);

  if (rawPressed != channelButtonRawState) {
    channelButtonRawState = rawPressed;
    channelButtonRawChangedMillis = now;
  }

  // Debounce raw input and only act on clean transitions.
  if ((now - channelButtonRawChangedMillis) < 40) {
    return;
  }

  bool pressed = channelButtonRawState;
  if (!pressed) {
    if (!lastChannelButtonPressed) {
      channelButtonReleaseStableMillis = now;
    }
    // Rearm only after button has been released and stable for a bit.
    if ((now - channelButtonReleaseStableMillis) > 180) {
      channelButtonLatched = false;
    }
  }

  if (pressed && !channelButtonLatched && (now - lastChannelButtonMillis) > 700) {
    channelButtonLatched = true;
    lastChannelButtonMillis = now;
    Serial.printf("BOOT_BUTTON -> CHANNEL_UP %d -> %d\n", channel, (channel + 1) % channelData->getChannelCount());
    channelUp();
  }
  lastChannelButtonPressed = pressed;
#endif
}

void loop()
{
#ifdef HAS_IR_REMOTE
  RemoteCommands command = remoteInput->getLatestCommand();
  if (command != RemoteCommands::UNKNOWN)
  {
    switch (command)
    {
    case RemoteCommands::POWER:
      // log out RAM usage
      Serial.printf("Total heap: %d\n", ESP.getHeapSize());
      Serial.printf("Free heap: %d\n", ESP.getFreeHeap());
      Serial.printf("Total PSRAM: %d\n", ESP.getPsramSize());
      Serial.printf("Free PSRAM: %d\n", ESP.getFreePsram());

      videoPlayer->stop();
      display.drawTuningText(TUNING_SERVER_INFO);
      Serial.println("POWER");
      // get the channel info
      while(!channelData->fetchChannelData()) {
        Serial.println("Failed to fetch channel data");
        delay(1000);
      }
      videoPlayer->setChannel(0);
      videoPlayer->play();
      break;
    case RemoteCommands::VOLUME_UP:
      volumeUp();
      break;
    case RemoteCommands::VOLUME_DOWN:
      volumeDown();
      break;
    case RemoteCommands::CHANNEL_UP:
      channelUp();
      break;
    case RemoteCommands::CHANNEL_DOWN:
      channelDown();
      break;
    }
    delay(100);
    remoteInput->getLatestCommand();
  }
#endif
#ifdef HAS_BUTTONS
  if (buttonLeft()) {
    channelDown();
  }
  if (buttonRight()) {
    channelUp();
  }
  if (buttonUp()) {
    volumeUp();
  }
  if (buttonDown()) {
    volumeDown();
  }
  if (buttonPowerOff()) {
    Serial.println("POWER OFF");
    delay(500);
    powerDeepSeep();
  }
  handleChannelSwitchButton();
  buttonLoop();
#else
    handleChannelSwitchButton();
    // important this needs to stay otherwise we are constantly polling the IR Remote
    // and there's no time for anything else to run.
    delay(200);
#endif
}
