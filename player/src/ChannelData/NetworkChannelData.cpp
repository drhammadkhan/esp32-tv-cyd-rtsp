#include <Arduino.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>
#include "NetworkChannelData.h"

NetworkChannelData::NetworkChannelData(const char *channelInfoURL, const char *frameURL, const char *audioURL, const char *clientQuery)
    : mChannelInfoURL(channelInfoURL),
      mFrameURL(frameURL),
      mAudioURL(audioURL),
      mClientQuery(clientQuery ? clientQuery : "") {

}

bool NetworkChannelData::fetchChannelData() {
  // check to see if we are connected to Wifi
  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }
  // make a HTTP request to get the channel data
  HTTPClient http;
  http.setReuse(false);
  http.useHTTP10(true);
  http.setConnectTimeout(3000);
  http.setTimeout(3000);
  if (!http.begin(mChannelInfoURL.c_str())) {
    Serial.println("HTTP begin failed");
    return false;
  }
  int httpCode = http.GET();
  if (httpCode == HTTP_CODE_OK) {
    // read and parse the full JSON payload safely
    String response = http.getString();
    // parse the response
    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, response);
    if (error) {
      Serial.println("Failed to parse channel data");
      http.end();
      return false;
    }
    mChannelLengths.clear();
    // get the channel lengths
    for (int i=0; i<doc.size(); i++) {
      mChannelLengths.push_back(doc[i]);
    }
    bool ok = mChannelLengths.size() > 0;
    http.end();
    return ok;
  } else {
    Serial.printf("HTTP error: %d\n", httpCode);
    http.end();
    return false;
  }
}

std::string NetworkChannelData::getFrameURL() {
  return mFrameURL + "/" + std::to_string(mChannelNumber);
}

std::string NetworkChannelData::getAudioURL() {
  return mAudioURL + "/" + std::to_string(mChannelNumber);
}

std::string NetworkChannelData::getClientQuery() {
  return mClientQuery;
}

void NetworkChannelData::setChannel(int channel) {
  mChannelNumber = channel;
}
